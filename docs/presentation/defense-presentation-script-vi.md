# Kịch bản bảo vệ IMP: Evidence Before Leaderboards

**Thời lượng trình bày:** 11 phút, tổng cộng 660 giây.
**Tuyến trình bày mặc định:** S01-S10, S16, S17. Slides 11-15 dành cho Q&A, không tính giờ.
**Năm evidence lane bắt buộc tách biệt:** historical Paper RQ1 L191/L192, gồm historical Clean-v3 source-report audit và adaptive development-validation; Loop206 RQ2 matched train-screen; fixed L206 cache; live illustrative L206-control so với reconstructed L192; prospective RQ1-v2 verified admission, hiện vẫn bị blocked.

## Slide 1 - Evidence before leaderboard claims (35 giây)
### Mục tiêu
Đặt luận đề: giá trị khoa học đến từ bằng chứng đúng phạm vi, không chỉ từ điểm số cao.
### Lời nói
"Kính thưa thầy cô, đề tài so sánh hệ thống IMP MiT-B3 U-Net với nnU-Net v2 và kiểm tra một kênh contour bổ sung. Benchmark cũ có leakage, còn giả thuyết contour cho kết quả âm. Đóng góp của chúng tôi là sửa hợp đồng bằng chứng, báo cáo đúng phạm vi và giữ protected test đóng kín. Đây là nghiên cứu phi lâm sàng, không phải công cụ chẩn đoán hay tuyên bố state of the art."
### Hiểu sâu
Thông điệp "evidence before leaderboards" nhấn mạnh rằng điểm số chỉ có ý nghĩa khi split, protocol, metric và claim cùng hợp lệ.
### Chuyển ý
"Trước khi so sánh mô hình, cần xem vì sao bảng xếp hạng cũ không còn mang ý nghĩa khoa học."
### Không được nói
Không gọi hệ thống là công cụ chẩn đoán lâm sàng, state of the art hoặc mô hình tốt nhất.

## Slide 2 - Leakage changed the meaning of every favorable score (50 giây)
### Mục tiêu
Giải thích lỗi thiết kế split và lý do kết quả legacy chỉ còn giá trị lịch sử.
### Lời nói
"Audit legacy phát hiện 3 danh tính bệnh nhân và 13 dòng dữ liệu cắt qua ranh giới split. Khi cùng một danh tính xuất hiện ở cả train và evaluation, kết quả không còn được diễn giải như đánh giá độc lập theo bệnh nhân. Vì vậy, các giá trị test-v2 cũ chỉ được giữ như bằng chứng lịch sử có nhiễm, không được dùng để xếp hạng khoa học. Việc sửa lỗi bắt đầu từ identity group, exact duplicate, perceptual duplicate và split group, chứ không bắt đầu từ tuning mô hình."
### Hiểu sâu
Leakage làm sai validity vì train và evaluation có thể chứa thông tin tương quan cùng bệnh nhân, nên evaluation mất tính độc lập theo bệnh nhân dù phép tính metric vẫn đúng về mặt kỹ thuật.
### Chuyển ý
"Sau khi tách bằng chứng cũ, chúng tôi đặt hai câu hỏi khác nhau với hai contract khác nhau."
### Không được nói
Không mô tả điểm legacy như kết quả test độc lập; không nói audit chứng minh mọi quan hệ sinh học ẩn đều không tồn tại.

## Slide 3 - Two hypotheses, two evidence classes (40 giây)
### Mục tiêu
Phân tách RQ1 system comparison khỏi RQ2 controlled ablation.
### Lời nói
"RQ1 hỏi hai hệ thống hoàn chỉnh trade-off overlap, boundary, precision và recall như thế nào. RQ2 hỏi riêng liệu kênh contour ràng buộc saliency có cải thiện control MiT-B3 đã được match hay không. RQ1 dùng adaptive development-validation, chỉ có một run được ghi nhận cho mỗi hệ thống, nên kết luận chỉ mang tính mô tả. RQ2 dùng Loop206 train-screen với ba selected seeds; khả năng diễn giải nhân quả mạnh hơn vì chỉ kênh đầu vào thứ tư thay đổi. Quy tắc chung là từ chối mọi kết luận vượt quá partition, seed hoặc metric contract."
### Hiểu sâu
"Architecture" không đồng nghĩa với "system". Chỉ RQ2 có thiết kế one-factor đủ để quy thay đổi quan sát được cho intervention trong phạm vi protocol.
### Chuyển ý
"Pipeline tiếp theo là bản đồ các cổng kiểm tra trước khi output đi vào paper hoặc demo."
### Không được nói
Không gộp RQ1 và RQ2 thành một bảng xếp hạng; không gọi RQ1 là ablation riêng của encoder.

## Slide 4 - The claim can only be as strong as its weakest evidence boundary (35 giây)
### Mục tiêu
Dùng pipeline làm trục tương tác và trục lập luận.
### Lời nói
"Slide này không phải sơ đồ trang trí. Mọi claim phải đi qua data audit, preprocessing, model protocol, robust evaluation, Loop206 ablation và evidence-bound demo. Tuyến mặc định đi từ trái sang phải; khi hỏi đáp, tôi quay lại module phù hợp rồi nhấn Escape để trở về. Nếu split bị nhiễm hoặc test còn sealed, claim phải dừng tại ranh giới đó."
### Hiểu sâu
Cùng một mask có thể hợp lệ trong live illustrative lane nhưng không tự động trở thành quantitative paper evidence.
### Chuyển ý
"Ranh giới đầu tiên là dữ liệu và lịch sử sửa split Clean-v3."
### Không được nói
Không trình bày pipeline như một benchmark workflow đã được xác nhận end-to-end.

## Slide 5 - Clean-v3 repairs identity leakage before model comparison (55 giây)
### Mục tiêu
Trình bày historical audit đúng phạm vi, đồng thời giữ prospective admission tách biệt.
### Lời nói
"Historical Clean-v3 source-report audit ghi nhận 2.869 ảnh: 2.008 train, 431 adaptive development-validation và 430 sealed test; audit không phát hiện identity-group overlap cắt qua split. Các kiểm tra gồm identity, exact duplicate, perceptual duplicate và split group. Cách nói chính xác là 'không phát hiện overlap', không phải chứng minh mọi quan hệ ẩn đều không tồn tại. Validation đã được mở và dùng adaptively; 430 ảnh test vẫn sealed. Prospective RQ1-v2 verified admission là lane riêng và hiện bị blocked; các số vừa nêu là historical metadata, không xác nhận prospective admission."
### Hiểu sâu
Historical audit cho phép diễn giải bằng chứng đã ghi nhận. Nó không thay thế hash-bound index và zero-crossing integrity report bắt buộc cho prospective RQ1-v2.
### Chuyển ý
"Trong phạm vi này, chúng tôi so sánh hai hệ thống hoàn chỉnh, không lập luận riêng về encoder."
### Không được nói
Không nói prospective RQ1-v2 đã admitted; không gọi validation là test; không mở 430 ảnh test.

## Slide 6 - The comparison is between complete systems, not isolated encoders (60 giây)
### Mục tiêu
Làm rõ system contract và geometry mismatch.
### Lời nói
"IMP control L191 dùng LAB-luminance CLAHE, percentile stretch, median filter và MiT-B3 U-Net trong pipeline cố định 384 x 384. nnU-Net v2 L192 dùng raw RGB, generated plans và pipeline 2D self-configuring 256 x 256. Decoder, loss, augmentation, schedule, postprocessing, selection policy và geometry đều khác; output L192 còn được khôi phục từ 256 lên metric canvas 384. Vì vậy, Paper RQ1 là so sánh cấp hệ thống với một strong baseline, nhưng chỉ cung cấp bằng chứng mô tả với suy luận nhân quả yếu; đây không phải controlled architecture ablation. Chênh lệch có thể đến từ nhiều thành phần, đặc biệt từ boundary localization, nên không được quy riêng cho encoder."
### Hiểu sâu
Một so sánh mạnh hơn cần dùng cùng identities, original-image geometry, conditions, metrics, tuning budget, seeds và hardware accounting.
### Chuyển ý
"Vậy các con số đã ghi nhận được phép hỗ trợ kết luận đến mức nào?"
### Không được nói
Không nói nnU-Net "superior"; không quy kết quả riêng cho MiT-B3 hoặc raw RGB.

## Slide 7 - nnU-Net records higher point estimates, not statistical superiority (55 giây)
### Mục tiêu
Báo cáo chính xác bốn metric và mức bất định của RQ1.
### Lời nói
"Trên adaptive development-validation, IMP có robust Dice 0.8959, Precision 0.9088, Recall 0.9128 và boundary F1 0.4145. nnU-Net có robust Dice 0.9019, Precision 0.9056, Recall 0.9246 và boundary F1 0.4369. nnU-Net ghi nhận point estimate cao hơn ở robust Dice, boundary F1 và Recall; IMP giữ Precision cao hơn. Đây là trade-off giữa các metric, không phải kết luận một hệ thống thắng tuyệt đối. Không có paired confidence interval; kết quả thuộc older-geometry, single-run, selection-optimistic và contract-specific. Protected test chưa được mở."
### Hiểu sâu
"Descriptive" nghĩa là ghi lại hướng quan sát trong contract hiện có, không ước lượng certainty trên dữ liệu chưa thấy.
### Chuyển ý
"Để kiểm tra một cơ chế cụ thể chặt chẽ hơn, chúng tôi chuyển sang matched Loop206."
### Không được nói
Không nói statistically significant, superior, test performance, generalization hoặc SOTA.

## Slide 8 - Loop206 isolates one intervention: the fourth input channel (60 giây)
### Mục tiêu
Mô tả thiết kế RQ2 và phạm vi của interval.
### Lời nói
"Loop206 có 308 fit groups và 76 train-screen holdout groups. Control và candidate dùng chung architecture, preprocessing, ba selected seeds, 20 epochs cho mỗi arm, budget và loss; khác biệt duy nhất là zero channel so với contour channel thứ tư. Ba preregistered views được đánh giá, sau đó cluster bootstrap resample theo group sau khi average seeds và views. Không có protected image hoặc mask nào tham gia fitting, thresholding, selection hay evaluation. Đây là phép kiểm tra nhân quả mạnh hơn RQ1, nhưng interval vẫn conditional on the three selected seeds; nó không ước lượng seed-selection uncertainty."
### Hiểu sâu
Bootstrap theo split group giữ đúng đơn vị phụ thuộc hơn việc resample từng ảnh; ba selected seeds vẫn là điều kiện cố định của estimand.
### Chuyển ý
"Kết quả của intervention duy nhất này là âm, và đó là phát hiện trung tâm của báo cáo."
### Không được nói
Không nói interval đại diện cho quần thể seed; không nói protected test đã xác nhận kết quả.

## Slide 9 - The contour channel fails the primary gate (55 giây)
### Mục tiêu
Nêu rõ negative finding, conditional interval và giới hạn audit.
### Lời nói
"Candidate trừ control cho robust Dice là -0.0313, với conditional 95% interval [-0.0491, -0.0156], hoàn toàn dưới không. Boundary F1 là -0.0147, interval [-0.0308, 0.0010]. Kết quả aggregate conditional on the three selected seeds là bằng chứng hiện có; hướng của từng seed chưa được thiết lập bởi bằng chứng paper được cấp quyền. Candidate bị reject trước protected evaluation để tránh protected-test fishing. Paper công bố Dice và boundary F1; các gate endpoint khác không được tự suy ra khi chưa hiển thị."
### Hiểu sâu
Negative result bác bỏ binary contour mechanism này trong protocol này; nó không bác bỏ mọi boundary method.
### Chuyển ý
"Bây giờ tôi chuyển sang demo; demo cũng phải tuân thủ evidence lane riêng."
### Không được nói
Không nói contour channel luôn có hại; không suy ra một gate chưa hiển thị bằng endpoint cụ thể. Deck và paper Results cùng đánh dấu per-seed directions `unavailable`; chỉ trình bày kết quả aggregate conditional.

## Slide 10 - Two demo lanes answer different questions (90 giây)
### Mục tiêu
Chạy dual-live an toàn; tách live illustrative khỏi fixed audited; có fail-closed fallback.
### Lời nói
**0-15 giây.** "Demo phi lâm sàng, chỉ minh họa. ground_truth_not_loaded: không tải ground truth, nên không có claim accuracy, generalization hay chẩn đoán. Test-v3 và PH2 vẫn sealed."

**15-30 giây.** "Tôi chọn một public/synthetic sample đã prechecked và xác nhận lane trước khi chạy."

**30-55 giây.** "Tôi gửi cùng RGB qua IMP L206-control-s206 trước, rồi reconstructed L192-nnUNet-v2-raw-100ep. Hai output dùng cùng đầu vào; đây không phải protected-test evidence."

**55-70 giây.** "Ba panel là original, IMP và nnU-Net. Receipt cho biết transaction identity và latency; nó không có nhãn tham chiếu nên không chứng minh accuracy."

**70-80 giây.** "Receipt chỉ được tạo sau khi cả hai arm hoàn thành. Nếu một arm lỗi, hệ thống không phát receipt hoàn chỉnh; release không có canonical live receipt."

**80-90 giây.** "Fixed L206 cache là một audited lane khác, so sánh zero channel với contour channel trên các mẫu cố định. Live output này không phải Paper RQ1."

**Nếu lỗi.** Giữ kết quả IMP hiện tại, xóa panel nnU-Net và receipt, rồi nói rõ hệ thống đang fail-closed. Nếu khôi phục runtime cần quá 20 giây, dừng live demo và chuyển sang saved static screenshot. Screenshot vẫn là minh họa phi lâm sàng, không có claim accuracy và không thay thế protected-test evidence.
### Hiểu sâu
Release manifest ghi nhận cả hai public samples có trong Clean-v3 training của L192, nằm ngoài L206 308-group fit và thuộc 76-group train-screen holdout; provenance độc lập ngoài manifest chưa được xác minh. Live illustrative lane chỉ trả lời runtime hiện hành hiển thị gì trên cùng RGB; fixed L206 cache trả lời zero channel và contour channel khác nhau ra sao trên các mẫu cố định đã được audit. Receipt chỉ chứng minh transaction identity và latency. Không loaded ground truth nghĩa là không có accuracy evidence; test-v3 và PH2 vẫn sealed. Runtime/browser acceptance bị blocked, nên offline deck là fallback chính.
### Chuyển ý
"Slides 11-15 dành cho hỏi đáp; tuyến chính chuyển thẳng sang giới hạn reproducibility."
### Không được nói
Không hiện ground truth; mask đẹp không chứng minh đúng. Không phát receipt nếu một arm chưa hoàn thành. Không gọi reconstructed runtime tương đương original runtime.

## Slide 16 - Artifacts are hash-bound; full experiments are not clone-runnable (60 giây)
### Mục tiêu
Phân tách digest, strict local audit, reconstructed runtime và operational acceptance.
### Lời nói
"Các claim đã ghi nhận được digest-recorded. Bản historical/superseded source-report strict audit chỉ xác minh exact bytes và artifact identity của manifest, tables, figures, PDF, model IDs cùng available audited reports trong lần audit đó; đây không phải current-release claim. Hash chỉ ràng buộc exact bytes của checkpoint hoặc artifact được liệt kê, không bind toàn bộ runtime và không chứng minh tính đúng khoa học. Ràng buộc bytes này không làm L191/L192 clone-runnable vì historical training code, configs, paired predictions và một số historical training reports vẫn thiếu. Runtime L192 là reconstructed; startup manifest và checks nhận diện cấu hình runtime cùng model identity hiện hành, không chứng minh original-runtime equivalence. Prospective RQ1-v2 vẫn pending/unverified; browser/runtime acceptance chưa established; protected test vẫn sealed."
### Hiểu sâu
Hash chỉ ràng buộc exact bytes của checkpoint hoặc artifact được liệt kê; startup manifest và checks nhận diện cấu hình runtime cùng model identity hiện hành. Không cơ chế nào bind toàn bộ runtime hoặc chứng minh accuracy, fairness, clinical validity, runtime equivalence, deploy readiness hay deterministic browser acceptance.
### Chuyển ý
"Vì vậy, kết luận cuối cùng hẹp hơn nhưng mạnh hơn về mặt khoa học."
### Không được nói
Không nói full reproduction, E2E acceptance, original-runtime equivalence hoặc prospective RQ1-v2 đã hoàn tất.

## Slide 17 - A narrower conclusion is the stronger scientific result (65 giây)
### Mục tiêu
Kết luận bằng bốn phát hiện và next experiment được phép đề xuất.
### Lời nói
"Tôi kết lại bằng bốn điểm. Một, phải repair leakage trước khi optimize; điểm cũ không cứu được split không hợp lệ. Hai, nnU-Net ghi nhận point estimate cao hơn trên adaptive validation, nhưng đây chỉ là complete-system comparison dưới legacy geometry contract. Ba, hard contour prior thất bại primary matched train-screen gate: robust Dice thay đổi -0.0313, conditional on the three selected seeds. Bốn, không có protected-test hoặc SOTA claim nào được cấp phép. Bước tiếp theo, sau verified admission, là chạy ba independent seeds cho mỗi complete-system arm trên cùng original-image geometry và metric code, xác thực sáu receipts, báo cáo chín crossed contrasts, rồi mới xét protected test."
### Hiểu sâu
Kết quả âm và claim hẹp vẫn là thông tin khoa học có giá trị vì chúng ngăn một leaderboard không được bằng chứng hỗ trợ.
### Chuyển ý
"Tôi sẵn sàng trả lời câu hỏi về evidence boundaries, matched Loop206 ablation hoặc các demo lanes."
### Không được nói
Không cam kết kết quả của future experiment; không mở protected test chỉ để "xác nhận" phát hiện mong muốn.

# Slides 11-15: ghi chú học cho Q&A, không tính giờ

## Slide 11 - Can a clean score survive an invalid split?
- **Vấn đề:** Leakage làm validation có vẻ độc lập hơn thực tế.
- **Phản ứng của dự án:** Audit identity group và split assignment trước khi diễn giải score; legacy có 3 danh tính bệnh nhân và 13 rows cắt qua boundary.
- **Giới hạn còn lại:** Audit giảm leakage có thể phát hiện, không chứng minh mọi quan hệ ẩn đều vắng mặt.
- **Trả lời ngắn:** "Score đẹp không cứu được split không hợp lệ; tôi chỉ nói historical Clean-v3 audit không phát hiện overlap."
- **Question-bank IDs:** Q1, Q2, Q20, Q21.

## Slide 12 - What makes a comparison fair enough to interpret?
- **Vấn đề:** Complete systems khác preprocessing, resolution, geometry, loss và policy.
- **Phản ứng của dự án:** Gọi RQ1 là descriptive system comparison; dùng matched Loop206 arms cho one-factor contour test.
- **Giới hạn còn lại:** Geometry 256-to-384 và nhiều khác biệt trong RQ1 ngăn component attribution.
- **Trả lời ngắn:** "nnU-Net là strong baseline, không phải controlled encoder ablation."
- **Question-bank IDs:** Q3, Q4, Q9.

## Slide 13 - How much certainty can adaptive validation support?
- **Vấn đề:** Adaptive validation và selected seeds dễ dẫn đến certainty quá mức.
- **Phản ứng của dự án:** Báo cáo RQ1 bằng point estimates; Loop206 dùng split-group bootstrap và ghi rõ conditional on the three selected seeds.
- **Giới hạn còn lại:** RQ1 không có paired CI; Loop206 không ước lượng seed-population uncertainty.
- **Trả lời ngắn:** "Bằng chứng hỗ trợ hướng quan sát trong contract hiện có, không hỗ trợ superiority hoặc certainty trên dữ liệu chưa thấy."
- **Question-bank IDs:** Q2, Q5, Q6.

## Slide 14 - When should an interactive demo be trusted?
- **Vấn đề:** Visual output dễ bị nhầm thành accuracy hoặc generalization evidence.
- **Phản ứng của dự án:** Tách live illustrative lane khỏi fixed L206 cache; không tải ground truth; chỉ phát receipt sau dual success; lỗi thì fail-closed.
- **Giới hạn còn lại:** Không có canonical live receipt; reconstructed runtime chưa được chứng minh tương đương original runtime; browser acceptance bị blocked.
- **Trả lời ngắn:** "Demo chỉ chứng minh đúng lane, input và transaction đã hiển thị; mask đẹp không chứng minh đúng hoặc chính xác."
- **Question-bank IDs:** Q10, Q11, Q12, Q13, Q14, Q15, Q17, Q23.

## Slide 15 - What can a reproducible deployment actually prove?
- **Vấn đề:** Hash-bound artifact không đồng nghĩa với clone-runnable experiment.
- **Phản ứng của dự án:** Strict local audit xác minh exact bytes, manifests và available audited reports; deployment identity được tách khỏi historical experiment reproduction.
- **Giới hạn còn lại:** Missing historical training reports, data, weights, caches, code/configs và paired predictions ngăn tái chạy đầy đủ; prospective RQ1-v2 admission còn blocked.
- **Trả lời ngắn:** "Lab khác có thể audit tracked artifacts và available audited reports, nhưng chưa thể rerun đầy đủ historical experiments từ fresh clone vì các historical training reports và inputs cần thiết còn thiếu."
- **Question-bank IDs:** Q16, Q18, Q19, Q22, Q24.

# Rehearsal checklist

- [ ] Offline deck mở được; Slides 1-10, 16 và 17 đi đúng presenter route.
- [ ] Kiểm tra local URL http://127.0.0.1:7860 và Cloudflare URL nếu có; chỉ demo URL đã qua preflight.
- [ ] Chọn một public/synthetic sample đã prechecked; không hiện ground truth.
- [ ] Saved static screenshot sẵn sàng nếu recovery vượt 20 giây.
- [ ] Non-clinical warning luôn hiển thị; nói "not a diagnosis" trước thao tác live.
- [ ] Thuộc đúng năm lane: historical Paper RQ1 L191/L192; Loop206 RQ2; fixed L206 cache; live L206-control so với reconstructed L192; prospective RQ1-v2.
- [ ] Nói rõ protected test-v3 và PH2 vẫn sealed; demo không có accuracy claim.
- [ ] Đặt hard stop ở 11 phút; bỏ qua Slides 11-15 nếu chưa vào Q&A.
