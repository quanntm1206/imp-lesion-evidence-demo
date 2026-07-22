from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
from xml.etree import ElementTree as ET
import warnings
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
INJECTOR_PATH = ROOT / "scripts" / "presentation" / "inject_pptx_navigation.py"


def _load_injector():
    spec = importlib.util.spec_from_file_location("pptx_navigation", INJECTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
SLIDE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
LAYOUT_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout"
HYPERLINK_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
OFFICE_DOCUMENT_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)
PRESENTATION_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)
SLIDE_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
)
SLIDE_LAYOUT_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"
)
SLIDE_MASTER_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"
)
THEME_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.theme+xml"
JUMP_ACTION = "ppaction://hlinksldjump"
SLIDE_COUNT = 17


def _shape_xml(index: int, name: str, relationship_id: str | None = None) -> str:
    click = (
        f'<a:hlinkClick r:id="{relationship_id}" action="{JUMP_ACTION}"/>'
        if relationship_id
        else ""
    )
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{index}" name="{name}">{click}</p:cNvPr>'
        "<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
        "<p:spPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"1\" cy=\"1\"/></a:xfrm>"
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:sp>'
    )


def _slide_xml(names: list[str], *, click_kind: str | None = None) -> bytes:
    shapes = "".join(
        _shape_xml(
            index,
            name,
            "rIdExisting" if name == "pipeline-node-0" and click_kind else None,
        )
        for index, name in enumerate(names, start=2)
    )
    return (
        f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}" '
        f'xmlns:mc="{MC}" xmlns:p14="{P14}" mc:Ignorable="p14">'
        "<p:cSld><p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
        "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        f"{shapes}</p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    ).encode("utf-8")


def _relationships_xml(
    *,
    click_kind: str | None = None,
    duplicate_id: bool = False,
    invalid_id: bool = False,
    unrelated_external: bool = False,
) -> bytes:
    layout_id = "1invalid" if invalid_id else "rIdLayout"
    relationships = [
        f'<Relationship Id="{layout_id}" Type="{LAYOUT_REL}" '
        'Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    if duplicate_id:
        relationships.append(
            f'<Relationship Id="{layout_id}" Type="{SLIDE_REL}" Target="slide2.xml"/>'
        )
    if click_kind == "conflicting":
        relationships.append(
            f'<Relationship Id="rIdExisting" Type="{SLIDE_REL}" Target="slide6.xml"/>'
        )
    if click_kind == "external":
        relationships.append(
            f'<Relationship Id="rIdExisting" Type="{HYPERLINK_REL}" '
            'Target="https://example.invalid/" TargetMode="External"/>'
        )
    if unrelated_external:
        relationships.append(
            f'<Relationship Id="rIdExternal" Type="{HYPERLINK_REL}" '
            'Target="https://example.invalid/" TargetMode="External"/>'
        )
    return (
        f'<Relationships xmlns="{PACKAGE_REL}">{"".join(relationships)}</Relationships>'
    ).encode("utf-8")


def _content_types_xml() -> bytes:
    overrides = [
        f'<Override PartName="/ppt/presentation.xml" ContentType="{PRESENTATION_CONTENT_TYPE}"/>',
        f'<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="{SLIDE_LAYOUT_CONTENT_TYPE}"/>',
        f'<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="{SLIDE_MASTER_CONTENT_TYPE}"/>',
        f'<Override PartName="/ppt/theme/theme1.xml" ContentType="{THEME_CONTENT_TYPE}"/>',
    ] + [
        f'<Override PartName="/ppt/slides/slide{number}.xml" '
        f'ContentType="{SLIDE_CONTENT_TYPE}"/>'
        for number in range(1, SLIDE_COUNT + 1)
    ]
    return (
        f'<Types xmlns="{CONTENT_TYPES}">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(overrides)}</Types>'
    ).encode("utf-8")


def _presentation_xml() -> bytes:
    slides = "".join(
        f'<p:sldId id="{255 + number}" r:id="rIdSlide{number}"/>'
        for number in range(1, SLIDE_COUNT + 1)
    )
    return (
        f'<p:presentation xmlns:p="{P}" xmlns:r="{R}">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rIdMaster"/>'
        "</p:sldMasterIdLst>"
        f"<p:sldIdLst>{slides}</p:sldIdLst>"
        '<p:sldSz cx="12192000" cy="6858000"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        "</p:presentation>"
    ).encode("utf-8")


def _presentation_relationships_xml(order: list[int] | None = None) -> bytes:
    target_order = order or list(range(1, SLIDE_COUNT + 1))
    relationships = (
        '<Relationship Id="rIdMaster" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" '
        'Target="/ppt/slideMasters/slideMaster1.xml"/>'
        + "".join(
        f'<Relationship Id="rIdSlide{position}" Type="{SLIDE_REL}" '
        f'Target="/ppt/slides/slide{target}.xml"/>'
        for position, target in enumerate(target_order, start=1)
        )
    )
    return (
        f'<Relationships xmlns="{PACKAGE_REL}">{relationships}</Relationships>'
    ).encode("utf-8")


def _root_relationships_xml() -> bytes:
    return (
        f'<Relationships xmlns="{PACKAGE_REL}">'
        f'<Relationship Id="rId1" Type="{OFFICE_DOCUMENT_REL}" '
        'Target="/ppt/presentation.xml"/>'
        "</Relationships>"
    ).encode("utf-8")


def _layout_xml() -> bytes:
    return (
        f'<p:sldLayout xmlns:p="{P}" xmlns:a="{A}" type="blank" preserve="1">'
        '<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>"
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/>'
        '<a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>'
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
    ).encode("utf-8")


def _master_xml() -> bytes:
    return (
        f'<p:sldMaster xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>"
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/>'
        '<a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>'
        '<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" '
        'accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" '
        'bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rIdLayout"/>'
        "</p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/>"
        "<p:otherStyle/></p:txStyles></p:sldMaster>"
    ).encode("utf-8")


def _theme_xml() -> bytes:
    return (
        f'<a:theme xmlns:a="{A}" name="Minimal"><a:themeElements>'
        '<a:clrScheme name="Minimal"><a:dk1><a:sysClr val="windowText" lastClr="000000"/>'
        '</a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="1F497D"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/></a:lt2>'
        '<a:accent1><a:srgbClr val="4F81BD"/></a:accent1>'
        '<a:accent2><a:srgbClr val="C0504D"/></a:accent2>'
        '<a:accent3><a:srgbClr val="9BBB59"/></a:accent3>'
        '<a:accent4><a:srgbClr val="8064A2"/></a:accent4>'
        '<a:accent5><a:srgbClr val="4BACC6"/></a:accent5>'
        '<a:accent6><a:srgbClr val="F79646"/></a:accent6>'
        '<a:hlink><a:srgbClr val="0000FF"/></a:hlink>'
        '<a:folHlink><a:srgbClr val="800080"/></a:folHlink></a:clrScheme>'
        '<a:fontScheme name="Minimal"><a:majorFont><a:latin typeface="Arial"/>'
        '</a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont>'
        '</a:fontScheme><a:fmtScheme name="Minimal"><a:fillStyleLst/>'
        '<a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>'
        "</a:themeElements></a:theme>"
    ).encode("utf-8")


def _make_deck(
    path: Path,
    *,
    missing_member: str | None = None,
    missing_shape: tuple[int, str] | None = None,
    duplicate_member: str | None = None,
    duplicate_shape: tuple[int, str] | None = None,
    malformed_relationships: int | None = None,
    duplicate_relationship_id: int | None = None,
    invalid_relationship_id: int | None = None,
    click_kind: str | None = None,
    unrelated_external_slide: int | None = None,
    presentation_order: list[int] | None = None,
    extra_entries: dict[str, bytes] | None = None,
) -> None:
    entries = {
        "[Content_Types].xml": _content_types_xml(),
        "_rels/.rels": _root_relationships_xml(),
        "ppt/presentation.xml": _presentation_xml(),
        "ppt/_rels/presentation.xml.rels": _presentation_relationships_xml(
            presentation_order
        ),
        "ppt/slideLayouts/slideLayout1.xml": _layout_xml(),
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": (
            f'<Relationships xmlns="{PACKAGE_REL}"><Relationship Id="rIdMaster" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" '
            'Target="../slideMasters/slideMaster1.xml"/></Relationships>'
        ).encode("utf-8"),
        "ppt/slideMasters/slideMaster1.xml": _master_xml(),
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": (
            f'<Relationships xmlns="{PACKAGE_REL}"><Relationship Id="rIdLayout" '
            f'Type="{LAYOUT_REL}" Target="../slideLayouts/slideLayout1.xml"/>'
            '<Relationship Id="rIdTheme" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" '
            'Target="../theme/theme1.xml"/></Relationships>'
        ).encode("utf-8"),
        "ppt/theme/theme1.xml": _theme_xml(),
    }
    for number in range(1, SLIDE_COUNT + 1):
        names: list[str] = []
        if number == 4:
            names = [f"pipeline-node-{index}" for index in range(6)]
        if 5 <= number <= 10:
            names = ["back-to-pipeline"]
        if missing_shape and missing_shape[0] == number:
            names = [name for name in names if name != missing_shape[1]]
        if duplicate_shape and duplicate_shape[0] == number:
            names.append(duplicate_shape[1])
        slide_click = click_kind if number == 4 else None
        entries[f"ppt/slides/slide{number}.xml"] = _slide_xml(
            names, click_kind=slide_click
        )
        entries[f"ppt/slides/_rels/slide{number}.xml.rels"] = (
            b"<invalid/>"
            if number == malformed_relationships
            else _relationships_xml(
                click_kind=slide_click,
                duplicate_id=number == duplicate_relationship_id,
                invalid_id=number == invalid_relationship_id,
                unrelated_external=number == unrelated_external_slide,
            )
        )
    if extra_entries:
        entries.update(extra_entries)
    if missing_member:
        entries.pop(missing_member)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        for name, data in entries.items():
            package.writestr(name, data)
        if duplicate_member:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                package.writestr(duplicate_member, entries[duplicate_member])


def _mark_first_entry_encrypted(path: Path) -> None:
    data = bytearray(path.read_bytes())
    local = data.index(b"PK\x03\x04")
    central = data.index(b"PK\x01\x02")
    local_flags = struct.unpack_from("<H", data, local + 6)[0] | 1
    central_flags = struct.unpack_from("<H", data, central + 8)[0] | 1
    struct.pack_into("<H", data, local + 6, local_flags)
    struct.pack_into("<H", data, central + 8, central_flags)
    path.write_bytes(data)


def _xml(package: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(package.read(name))


def _relationship_target(package: zipfile.ZipFile, slide: int, shape_name: str) -> str:
    slide_xml = _xml(package, f"ppt/slides/slide{slide}.xml")
    shape = next(
        shape
        for shape in slide_xml.iter(f"{{{P}}}cNvPr")
        if shape.attrib.get("name") == shape_name
    )
    link = shape.find(f"{{{A}}}hlinkClick")
    assert link is not None
    relationship_id = link.attrib[f"{{{R}}}id"]
    relationships = _xml(package, f"ppt/slides/_rels/slide{slide}.xml.rels")
    relationship = next(
        relationship
        for relationship in relationships.findall(f"{{{PACKAGE_REL}}}Relationship")
        if relationship.attrib["Id"] == relationship_id
    )
    assert relationship.attrib["Type"] == SLIDE_REL
    assert "TargetMode" not in relationship.attrib
    assert link.attrib["action"] == JUMP_ACTION
    return relationship.attrib["Target"]


def _has_medium_fade(package: zipfile.ZipFile, slide: int) -> bool:
    slide_xml = _xml(package, f"ppt/slides/slide{slide}.xml")
    transition = slide_xml.find(f"{{{P}}}transition")
    return (
        transition is not None
        and transition.attrib.get("spd") == "med"
        and transition.find(f"{{{P}}}fade") is not None
    )


def _shape_names(package: zipfile.ZipFile, slide: int) -> set[str]:
    slide_xml = _xml(package, f"ppt/slides/slide{slide}.xml")
    return {
        shape.attrib["name"]
        for shape in slide_xml.iter(f"{{{P}}}cNvPr")
        if "name" in shape.attrib
    }


def _navigation_items() -> list[tuple[int, str, str]]:
    pipeline_targets = [5, 6, 6, 7, 8, 10]
    return [
        (4, f"pipeline-node-{index}", f"slide{target}.xml")
        for index, target in enumerate(pipeline_targets)
    ] + [
        (number, "back-to-pipeline", "slide4.xml") for number in range(5, 11)
    ]


def test_injects_internal_pipeline_jumps_back_links_and_medium_fades(
    tmp_path: Path,
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source)

    injector.inject_navigation(source, output)

    with zipfile.ZipFile(output) as package:
        assert injector.SLIDE_COUNT == SLIDE_COUNT
        for slide, shape, target in _navigation_items():
            assert _relationship_target(package, slide, shape) == target
        assert all(
            _has_medium_fade(package, number)
            for number in range(1, SLIDE_COUNT + 1)
        )
        assert all(
            "back-to-pipeline" not in _shape_names(package, number)
            for number in range(11, 16)
        )


def test_preserves_markup_compatibility_namespace_prefixes(tmp_path: Path) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source)

    injector.inject_navigation(source, output)

    with zipfile.ZipFile(output) as package:
        slide = package.read("ppt/slides/slide1.xml")
    assert f'xmlns:mc="{MC}"'.encode() in slide
    assert f'xmlns:p14="{P14}"'.encode() in slide
    root = ET.fromstring(slide)
    assert root.attrib[f"{{{MC}}}Ignorable"] == "p14"


@pytest.mark.parametrize(
    "missing_member",
    [
        "[Content_Types].xml",
        "_rels/.rels",
        "ppt/presentation.xml",
        "ppt/_rels/presentation.xml.rels",
    ],
)
def test_rejects_missing_pptx_spine_member(
    tmp_path: Path, missing_member: str
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source, missing_member=missing_member)

    with pytest.raises(injector.NavigationInjectionError, match="PPTX spine"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_reordered_presentation_slide_targets(tmp_path: Path) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    order = [2, 1, *range(3, SLIDE_COUNT + 1)]
    _make_deck(source, presentation_order=order)

    with pytest.raises(injector.NavigationInjectionError, match="slide order"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_missing_stable_navigation_shape_without_writing_output(
    tmp_path: Path,
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source, missing_shape=(7, "back-to-pipeline"))

    with pytest.raises(injector.NavigationInjectionError, match="back-to-pipeline"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_duplicate_navigation_shape(tmp_path: Path) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source, duplicate_shape=(4, "pipeline-node-0"))

    with pytest.raises(injector.NavigationInjectionError, match="exactly one named shape"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_duplicate_zip_member(tmp_path: Path) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source, duplicate_member="ppt/slides/slide4.xml")

    with pytest.raises(injector.NavigationInjectionError, match="duplicate part names"):
        injector.inject_navigation(source, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("click_kind", "message"),
    [("conflicting", "conflicting click"), ("external", "external relationship")],
)
def test_rejects_conflicting_or_external_navigation_click(
    tmp_path: Path, click_kind: str, message: str
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source, click_kind=click_kind)

    with pytest.raises(injector.NavigationInjectionError, match=message):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_unrelated_external_relationship_without_writing_output(
    tmp_path: Path,
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source, unrelated_external_slide=17)

    with pytest.raises(injector.NavigationInjectionError, match="external relationship"):
        injector.inject_navigation(source, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("duplicate_slide", "invalid_slide"), [(4, None), (None, 7)]
)
def test_rejects_duplicate_or_invalid_existing_relationship_id(
    tmp_path: Path,
    duplicate_slide: int | None,
    invalid_slide: int | None,
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(
        source,
        duplicate_relationship_id=duplicate_slide,
        invalid_relationship_id=invalid_slide,
    )

    with pytest.raises(injector.NavigationInjectionError, match="relationship id"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_encrypted_zip_entry(tmp_path: Path) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source)
    _mark_first_entry_encrypted(source)

    with pytest.raises(injector.NavigationInjectionError, match="encrypted"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_member_exceeding_uncompressed_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source)
    monkeypatch.setattr(injector, "MAX_MEMBER_UNCOMPRESSED", 128)

    with pytest.raises(injector.NavigationInjectionError, match="member size"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_package_exceeding_total_uncompressed_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source)
    monkeypatch.setattr(injector, "MAX_TOTAL_UNCOMPRESSED", 1_024)

    with pytest.raises(injector.NavigationInjectionError, match="total size"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_rejects_excessive_zip_compression_ratio(tmp_path: Path) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source, extra_entries={"payload.bin": b"A" * 1_000_000})

    with pytest.raises(injector.NavigationInjectionError, match="compression ratio"):
        injector.inject_navigation(source, output)

    assert not output.exists()


def test_is_idempotent_for_every_navigation_shape_relationship_and_transition(
    tmp_path: Path,
) -> None:
    injector = _load_injector()
    source = tmp_path / "base.pptx"
    first = tmp_path / "first.pptx"
    second = tmp_path / "second.pptx"
    _make_deck(source)

    injector.inject_navigation(source, first)
    injector.inject_navigation(first, second)

    with zipfile.ZipFile(first) as first_package, zipfile.ZipFile(
        second
    ) as second_package:
        for slide, shape, target in _navigation_items():
            assert _relationship_target(first_package, slide, shape) == target
            assert _relationship_target(second_package, slide, shape) == target
        for number in range(1, SLIDE_COUNT + 1):
            assert _has_medium_fade(first_package, number)
            assert _has_medium_fade(second_package, number)
            first_rels = _xml(
                first_package, f"ppt/slides/_rels/slide{number}.xml.rels"
            )
            second_rels = _xml(
                second_package, f"ppt/slides/_rels/slide{number}.xml.rels"
            )
            first_ids = [relationship.attrib["Id"] for relationship in first_rels]
            second_ids = [relationship.attrib["Id"] for relationship in second_rels]
            assert len(first_ids) == len(set(first_ids))
            assert first_ids == second_ids


def test_rejects_malformed_deck_without_changing_source_or_existing_output(
    tmp_path: Path,
) -> None:
    injector = _load_injector()
    source = tmp_path / "malformed.pptx"
    output = tmp_path / "final.pptx"
    source.write_bytes(b"not a zip archive")
    output.write_bytes(b"keep-existing-output")
    source_before = source.read_bytes()

    with pytest.raises(injector.NavigationInjectionError, match="valid PPTX ZIP"):
        injector.inject_navigation(source, output)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == b"keep-existing-output"


def test_rejects_malformed_relationship_part_without_writing_output(
    tmp_path: Path,
) -> None:
    injector = _load_injector()
    source = tmp_path / "malformed.pptx"
    output = tmp_path / "final.pptx"
    _make_deck(source, malformed_relationships=4)

    with pytest.raises(injector.NavigationInjectionError, match="relationships root"):
        injector.inject_navigation(source, output)

    assert not output.exists()
