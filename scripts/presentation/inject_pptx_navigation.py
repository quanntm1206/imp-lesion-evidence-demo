from __future__ import annotations

import argparse
from io import BytesIO
import os
from pathlib import Path
import posixpath
import re
import sys
import tempfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import quoteattr
import zipfile


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
SLIDE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
OFFICE_DOCUMENT_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)
PRESENTATION_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)
SLIDE_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
)
JUMP_ACTION = "ppaction://hlinksldjump"

CONTENT_TYPES_PART = "[Content_Types].xml"
ROOT_RELS_PART = "_rels/.rels"
PRESENTATION_PART = "ppt/presentation.xml"
PRESENTATION_RELS_PART = "ppt/_rels/presentation.xml.rels"

MAX_MEMBER_UNCOMPRESSED = 64 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200.0
READ_CHUNK_SIZE = 64 * 1024

PIPELINE_TARGETS = {
    "pipeline-node-0": 5,
    "pipeline-node-1": 6,
    "pipeline-node-2": 6,
    "pipeline-node-3": 7,
    "pipeline-node-4": 8,
    "pipeline-node-5": 10,
}
BACK_SLIDES = range(5, 11)
SLIDE_COUNT = 17

RELATIONSHIP_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
REGISTERABLE_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
RESERVED_ET_PREFIX = re.compile(r"^ns\d+$")

for prefix, namespace in (("a", A), ("p", P), ("r", R), ("", PACKAGE_REL)):
    ET.register_namespace(prefix, namespace)


class NavigationInjectionError(ValueError):
    """The base deck cannot safely receive deterministic navigation."""


def _tag(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def _slide_part(number: int) -> str:
    return f"ppt/slides/slide{number}.xml"


def _rels_part(number: int) -> str:
    return f"ppt/slides/_rels/slide{number}.xml.rels"


def _parse_xml(data: bytes, part: str) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError as error:
        raise NavigationInjectionError(f"malformed OOXML part: {part}") from error


def _namespace_mappings(data: bytes, part: str) -> tuple[tuple[str, str], ...]:
    mappings: dict[str, str] = {}
    try:
        for _event, (prefix, namespace) in ET.iterparse(
            BytesIO(data), events=("start-ns",)
        ):
            prefix = prefix or ""
            existing = mappings.get(prefix)
            if existing is not None and existing != namespace:
                raise NavigationInjectionError(
                    f"{part} rebinds namespace prefix: {prefix or '(default)'}"
                )
            mappings[prefix] = namespace
    except ET.ParseError as error:
        raise NavigationInjectionError(f"malformed OOXML part: {part}") from error
    return tuple(mappings.items())


def _serialize_xml(
    root: ET.Element, namespace_mappings: tuple[tuple[str, str], ...]
) -> bytes:
    for prefix, namespace in namespace_mappings:
        if prefix == "xml":
            continue
        if prefix and (
            not REGISTERABLE_PREFIX.fullmatch(prefix)
            or RESERVED_ET_PREFIX.fullmatch(prefix)
        ):
            raise NavigationInjectionError(
                f"unsupported namespace prefix for safe round-trip: {prefix}"
            )
        ET.register_namespace(prefix, namespace)
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    missing: list[str] = []
    for prefix, namespace in namespace_mappings:
        if prefix == "xml":
            continue
        declaration = "xmlns" if not prefix else f"xmlns:{prefix}"
        pattern = rb"\s" + re.escape(declaration.encode("ascii")) + rb"\s*="
        if not re.search(pattern, data):
            missing.append(f" {declaration}={quoteattr(namespace)}")
    if not missing:
        return data
    declaration_end = data.find(b"?>")
    root_start = data.find(b"<", declaration_end + 2)
    root_end = data.find(b">", root_start)
    if root_start < 0 or root_end < 0:
        raise NavigationInjectionError("failed to serialize OOXML root")
    return data[:root_end] + "".join(missing).encode("utf-8") + data[root_end:]


def _require_root(root: ET.Element, expected: str, part: str, label: str) -> None:
    if root.tag != expected:
        raise NavigationInjectionError(f"{part} has an invalid {label} root")


def _validate_relationships_root(
    root: ET.Element, part: str
) -> dict[str, ET.Element]:
    _require_root(
        root,
        _tag(PACKAGE_REL, "Relationships"),
        part,
        "relationships",
    )
    relationships: dict[str, ET.Element] = {}
    for relationship in root:
        if relationship.tag != _tag(PACKAGE_REL, "Relationship"):
            raise NavigationInjectionError(
                f"{part} contains a non-relationship child"
            )
        relationship_id = relationship.attrib.get("Id", "")
        if not RELATIONSHIP_ID.fullmatch(relationship_id):
            raise NavigationInjectionError(
                f"{part} has an invalid relationship id: {relationship_id or '(missing)'}"
            )
        if relationship_id in relationships:
            raise NavigationInjectionError(
                f"{part} has a duplicate relationship id: {relationship_id}"
            )
        if not relationship.attrib.get("Type") or not relationship.attrib.get("Target"):
            raise NavigationInjectionError(
                f"{part} relationship {relationship_id} lacks Type or Target"
            )
        target_mode = relationship.attrib.get("TargetMode")
        if target_mode is not None and target_mode != "External":
            raise NavigationInjectionError(
                f"{part} relationship {relationship_id} has invalid TargetMode"
            )
        relationships[relationship_id] = relationship
    return relationships


def _resolve_relationship_target(source_part: str, target: str) -> str:
    if target.startswith("/"):
        resolved = posixpath.normpath(target.lstrip("/"))
    else:
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))
    if resolved == ".." or resolved.startswith("../"):
        raise NavigationInjectionError(
            f"relationship target escapes the PPTX package: {target}"
        )
    return resolved


def _validate_content_types(parts: dict[str, bytes]) -> None:
    root = _parse_xml(parts[CONTENT_TYPES_PART], CONTENT_TYPES_PART)
    _require_root(root, _tag(CONTENT_TYPES, "Types"), CONTENT_TYPES_PART, "types")
    overrides: dict[str, str] = {}
    for override in root.findall(_tag(CONTENT_TYPES, "Override")):
        part_name = override.attrib.get("PartName", "")
        content_type = override.attrib.get("ContentType", "")
        if not part_name or part_name in overrides:
            raise NavigationInjectionError(
                "PPTX spine has missing or duplicate content-type overrides"
            )
        overrides[part_name] = content_type
    expected = {"/ppt/presentation.xml": PRESENTATION_CONTENT_TYPE}
    expected.update(
        {
            f"/ppt/slides/slide{number}.xml": SLIDE_CONTENT_TYPE
            for number in range(1, SLIDE_COUNT + 1)
        }
    )
    for part_name, content_type in expected.items():
        if overrides.get(part_name) != content_type:
            raise NavigationInjectionError(
                f"PPTX spine content type mismatch: {part_name}"
            )


def _validate_spine(parts: dict[str, bytes]) -> None:
    _validate_content_types(parts)
    root_rels = _parse_xml(parts[ROOT_RELS_PART], ROOT_RELS_PART)
    root_relationships = _validate_relationships_root(root_rels, ROOT_RELS_PART)
    office_documents = [
        relationship
        for relationship in root_relationships.values()
        if relationship.attrib["Type"] == OFFICE_DOCUMENT_REL
        and "TargetMode" not in relationship.attrib
    ]
    if len(office_documents) != 1 or _resolve_relationship_target(
        "", office_documents[0].attrib["Target"]
    ) != PRESENTATION_PART:
        raise NavigationInjectionError("PPTX spine has an invalid office document target")

    presentation = _parse_xml(parts[PRESENTATION_PART], PRESENTATION_PART)
    _require_root(
        presentation,
        _tag(P, "presentation"),
        PRESENTATION_PART,
        "presentation",
    )
    slide_lists = presentation.findall(_tag(P, "sldIdLst"))
    if len(slide_lists) != 1:
        raise NavigationInjectionError("PPTX spine must contain one slide ID list")
    slide_ids = slide_lists[0].findall(_tag(P, "sldId"))
    if len(slide_ids) != SLIDE_COUNT:
        raise NavigationInjectionError(
            f"PPTX spine must contain exactly {SLIDE_COUNT} ordered slide IDs"
        )

    presentation_rels = _parse_xml(parts[PRESENTATION_RELS_PART], PRESENTATION_RELS_PART)
    relationships = _validate_relationships_root(
        presentation_rels, PRESENTATION_RELS_PART
    )
    slide_relationships = [
        relationship
        for relationship in relationships.values()
        if relationship.attrib["Type"] == SLIDE_REL
    ]
    if len(slide_relationships) != SLIDE_COUNT:
        raise NavigationInjectionError(
            f"PPTX spine must contain exactly {SLIDE_COUNT} slide relationships"
        )

    numeric_ids: list[int] = []
    referenced_relationship_ids: list[str] = []
    for number, slide_id in enumerate(slide_ids, start=1):
        try:
            numeric_id = int(slide_id.attrib["id"])
        except (KeyError, ValueError) as error:
            raise NavigationInjectionError("PPTX spine has an invalid slide ID") from error
        relationship_id = slide_id.attrib.get(_tag(R, "id"), "")
        relationship = relationships.get(relationship_id)
        expected_target = _slide_part(number)
        if (
            relationship is None
            or relationship.attrib["Type"] != SLIDE_REL
            or "TargetMode" in relationship.attrib
            or _resolve_relationship_target(
                PRESENTATION_PART, relationship.attrib["Target"]
            )
            != expected_target
        ):
            raise NavigationInjectionError(
                f"PPTX spine slide order mismatch at slide {number}"
            )
        numeric_ids.append(numeric_id)
        referenced_relationship_ids.append(relationship_id)
    if (
        len(numeric_ids) != len(set(numeric_ids))
        or numeric_ids != sorted(numeric_ids)
        or any(number < 256 for number in numeric_ids)
        or len(referenced_relationship_ids) != len(set(referenced_relationship_ids))
        or set(referenced_relationship_ids)
        != {relationship.attrib["Id"] for relationship in slide_relationships}
    ):
        raise NavigationInjectionError("PPTX spine slide IDs are not unique and ordered")


def _validate_all_relationship_parts(parts: dict[str, bytes]) -> None:
    for part in sorted(name for name in parts if name.endswith(".rels")):
        root = _parse_xml(parts[part], part)
        relationships = _validate_relationships_root(root, part)
        if any(
            relationship.attrib.get("TargetMode") == "External"
            for relationship in relationships.values()
        ):
            raise NavigationInjectionError(f"{part} contains an external relationship")


def _shape(root: ET.Element, name: str, part: str) -> ET.Element:
    matches = [
        element
        for element in root.iter(_tag(P, "cNvPr"))
        if element.attrib.get("name") == name
    ]
    if len(matches) != 1:
        raise NavigationInjectionError(
            f"{part} must contain exactly one named shape: {name}"
        )
    return matches[0]


def _relationship_by_id(root: ET.Element, relationship_id: str) -> ET.Element:
    matches = [
        relationship
        for relationship in root.findall(_tag(PACKAGE_REL, "Relationship"))
        if relationship.attrib.get("Id") == relationship_id
    ]
    if len(matches) != 1:
        raise NavigationInjectionError(f"relationship id is not unique: {relationship_id}")
    return matches[0]


def _next_relationship_id(root: ET.Element) -> str:
    existing = {
        relationship.attrib["Id"]
        for relationship in root.findall(_tag(PACKAGE_REL, "Relationship"))
    }
    index = 1
    while f"rIdNav{index}" in existing:
        index += 1
    return f"rIdNav{index}"


def _slide_relationship(root: ET.Element, target_slide: int) -> ET.Element:
    target = f"slide{target_slide}.xml"
    matches = [
        relationship
        for relationship in root.findall(_tag(PACKAGE_REL, "Relationship"))
        if relationship.attrib.get("Type") == SLIDE_REL
        and relationship.attrib.get("Target") == target
        and "TargetMode" not in relationship.attrib
    ]
    if len(matches) > 1:
        raise NavigationInjectionError(
            f"duplicate internal slide relationships for target: {target}"
        )
    if matches:
        return matches[0]
    return ET.SubElement(
        root,
        _tag(PACKAGE_REL, "Relationship"),
        {"Id": _next_relationship_id(root), "Type": SLIDE_REL, "Target": target},
    )


def _link_shape(
    slide_root: ET.Element,
    relationships_root: ET.Element,
    shape_name: str,
    target_slide: int,
    slide_part: str,
) -> None:
    shape = _shape(slide_root, shape_name, slide_part)
    expected = _slide_relationship(relationships_root, target_slide)
    links = shape.findall(_tag(A, "hlinkClick"))
    if len(links) > 1:
        raise NavigationInjectionError(
            f"{slide_part} shape has multiple click actions: {shape_name}"
        )
    if links:
        link = links[0]
        relationship_id = link.attrib.get(_tag(R, "id"))
        if relationship_id is None:
            raise NavigationInjectionError(
                f"{slide_part} shape click action has no relationship id: {shape_name}"
            )
        relationship = _relationship_by_id(relationships_root, relationship_id)
        if (
            relationship.attrib.get("Id") != expected.attrib["Id"]
            or link.attrib.get("action") != JUMP_ACTION
        ):
            raise NavigationInjectionError(
                f"{slide_part} shape has a conflicting click action: {shape_name}"
            )
        return
    ET.SubElement(
        shape,
        _tag(A, "hlinkClick"),
        {_tag(R, "id"): expected.attrib["Id"], "action": JUMP_ACTION},
    )


def _set_medium_fade(slide_root: ET.Element, part: str) -> None:
    transitions = slide_root.findall(_tag(P, "transition"))
    if len(transitions) > 1:
        raise NavigationInjectionError(f"{part} contains multiple transitions")
    if transitions:
        slide_root.remove(transitions[0])
    transition = ET.Element(_tag(P, "transition"), {"spd": "med"})
    ET.SubElement(transition, _tag(P, "fade"))
    children = list(slide_root)
    index = 0
    for child in children:
        if child.tag in {_tag(P, "cSld"), _tag(P, "clrMapOvr")}:
            index += 1
        else:
            break
    slide_root.insert(index, transition)


def _validate_zip_metadata(infos: list[zipfile.ZipInfo]) -> None:
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise NavigationInjectionError("PPTX ZIP contains duplicate part names")
    total_size = 0
    for info in infos:
        if info.flag_bits & 0x1:
            raise NavigationInjectionError(f"PPTX ZIP entry is encrypted: {info.filename}")
        if info.file_size > MAX_MEMBER_UNCOMPRESSED:
            raise NavigationInjectionError(
                f"PPTX ZIP member size exceeds limit: {info.filename}"
            )
        total_size += info.file_size
        if total_size > MAX_TOTAL_UNCOMPRESSED:
            raise NavigationInjectionError("PPTX ZIP total size exceeds limit")
        if info.file_size:
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > MAX_COMPRESSION_RATIO:
                raise NavigationInjectionError(
                    f"PPTX ZIP compression ratio exceeds limit: {info.filename}"
                )


def _bounded_read(package: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    data = bytearray()
    with package.open(info) as stream:
        while True:
            chunk = stream.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_MEMBER_UNCOMPRESSED:
                raise NavigationInjectionError(
                    f"PPTX ZIP member size exceeds limit: {info.filename}"
                )
    if len(data) != info.file_size:
        raise NavigationInjectionError(
            f"PPTX ZIP member size mismatch: {info.filename}"
        )
    return bytes(data)


def _read_package(source: Path) -> tuple[list[zipfile.ZipInfo], dict[str, bytes]]:
    try:
        with zipfile.ZipFile(source) as package:
            infos = package.infolist()
            _validate_zip_metadata(infos)
            required = {
                CONTENT_TYPES_PART,
                ROOT_RELS_PART,
                PRESENTATION_PART,
                PRESENTATION_RELS_PART,
                *(
                    part
                    for number in range(1, SLIDE_COUNT + 1)
                    for part in (_slide_part(number), _rels_part(number))
                ),
            }
            names = {info.filename for info in infos}
            missing = sorted(required.difference(names))
            if missing:
                raise NavigationInjectionError(
                    f"PPTX spine is missing required parts: {', '.join(missing)}"
                )
            parts = {info.filename: _bounded_read(package, info) for info in infos}
    except NavigationInjectionError:
        raise
    except (zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError) as error:
        raise NavigationInjectionError("input is not a valid PPTX ZIP") from error
    _validate_all_relationship_parts(parts)
    _validate_spine(parts)
    return infos, parts


def inject_navigation(source: Path, output: Path) -> None:
    """Write a new PPTX with deterministic internal links and medium fades."""
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise NavigationInjectionError("input and output PPTX paths must differ")
    if not source.is_file():
        raise NavigationInjectionError(f"input PPTX does not exist: {source}")

    infos, parts = _read_package(source)
    for number in range(1, SLIDE_COUNT + 1):
        slide_part = _slide_part(number)
        rels_part = _rels_part(number)
        slide_namespaces = _namespace_mappings(parts[slide_part], slide_part)
        rels_namespaces = _namespace_mappings(parts[rels_part], rels_part)
        slide_root = _parse_xml(parts[slide_part], slide_part)
        relationships_root = _parse_xml(parts[rels_part], rels_part)
        _require_root(slide_root, _tag(P, "sld"), slide_part, "slide")
        _validate_relationships_root(relationships_root, rels_part)
        if number == 4:
            for shape_name, target_slide in PIPELINE_TARGETS.items():
                _link_shape(
                    slide_root,
                    relationships_root,
                    shape_name,
                    target_slide,
                    slide_part,
                )
        if number in BACK_SLIDES:
            _link_shape(
                slide_root,
                relationships_root,
                "back-to-pipeline",
                4,
                slide_part,
            )
        _set_medium_fade(slide_root, slide_part)
        _validate_relationships_root(relationships_root, rels_part)
        parts[slide_part] = _serialize_xml(slide_root, slide_namespaces)
        parts[rels_part] = _serialize_xml(relationships_root, rels_namespaces)

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w") as package:
            for info in infos:
                package.writestr(info, parts[info.filename])
        os.replace(temporary, output)
    except OSError as error:
        raise NavigationInjectionError(f"failed to write output PPTX: {output}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inject deterministic internal PPTX navigation and medium fades."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        inject_navigation(args.input, args.output)
    except NavigationInjectionError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
