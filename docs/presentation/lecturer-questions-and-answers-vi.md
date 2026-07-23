# Câu hỏi giảng viên và trả lời bảo vệ IMP

Tài liệu này dùng cho hỏi đáp có giới hạn bằng chứng. Năm lane phải luôn được tách riêng: historical Paper RQ1 L191/L192, Loop206 RQ2 matched train-screen, fixed L206 cache, live illustrative L206-control so với reconstructed L192, và prospective RQ1-v2 verified admission. Các câu trả lời không cấp phép claim về superiority, accuracy live, fairness, clinical validity, deploy-readiness, SOTA hoặc original-runtime equivalence.

## A. Research framing and contribution

### Q1 [P1] - Đóng góp khoa học chính của đề tài là gì nếu không có mô hình mới đạt SOTA?
**Trả lời ngắn:** Đóng góp chính là sửa hợp đồng bằng chứng trước khi xếp hạng mô hình: phát hiện leakage trong split cũ, tách một complete-system comparison khỏi một one-factor ablation, và báo cáo trung thực negative result của contour channel. Giá trị nằm ở claim đúng phạm vi, không nằm ở việc tuyên bố một hệ thống thắng tuyệt đối.

**Giải thích sâu:** Đề tài không tự nhận phát minh một kiến trúc segmentation mới. Historical Paper RQ1 ghi lại trade-off giữa hai hệ thống hoàn chỉnh trên adaptive development-validation; Loop206 RQ2 kiểm tra riêng kênh contour thứ tư trong một thiết kế matched. Khi contamination, geometry, seed và partition được nêu rõ, một kết luận hẹp có giá trị hơn leaderboard không hợp lệ. **Hỏi tiếp dễ gặp:** “Vậy chỉ sửa quy trình thì có phải đóng góp ML không?” **Trả lời:** Có, nếu quy trình làm thay đổi câu hỏi nào được phép trả lời, ngăn leakage và tạo controlled negative evidence; tuy nhiên không được đổi tên đóng góp này thành novelty kiến trúc.

**Bằng chứng:** Slides 1, 3 và 17; lanes historical Paper RQ1 và Loop206 RQ2; artifact class: approved defense script và historical source-report audit.

**Bẫy cần tránh:** Gọi IMP là SOTA, mô hình tốt nhất, hoặc công cụ chẩn đoán chỉ vì quy trình bằng chứng chặt hơn.

### Q2 [P2] - Vì sao RQ1 và RQ2 phải được xem là hai câu hỏi khác nhau?
**Trả lời ngắn:** RQ1 hỏi trade-off giữa hai complete systems L191 và L192, trong khi RQ2 hỏi một can thiệp duy nhất: zero channel so với contour channel thứ tư trong Loop206. RQ1 chỉ mang tính mô tả; RQ2 có khả năng quy kết trong phạm vi protocol tốt hơn vì các yếu tố còn lại được match.

**Giải thích sâu:** Paper RQ1 thay đổi đồng thời preprocessing, architecture, loss, augmentation, schedule, postprocessing, selection policy và geometry, nên không cô lập một cơ chế. Loop206 giữ control và candidate cùng architecture, preprocessing, loss, budget, epochs và selected seeds; chỉ input channel thứ tư thay đổi. Vì vậy, mức mạnh của suy luận khác nhau: RQ1 mô tả hướng point estimate của hệ thống; RQ2 kiểm tra tác động của binary contour mechanism trong đúng train-screen contract.

**Bằng chứng:** Slides 3, 6 và 8; lanes historical Paper RQ1 và Loop206 RQ2; artifact class: paper experiment protocol và approved defense script.

**Bẫy cần tránh:** Gộp RQ1 và RQ2 thành một leaderboard hoặc gọi RQ1 là controlled encoder ablation.

### Q3 [P2] - Một negative result như Loop206 có giá trị gì?
**Trả lời ngắn:** Nó cho biết contour channel được kiểm tra không vượt qua primary matched train-screen gate trong protocol đã khóa. Kết quả này ngăn nhóm tiếp tục quảng bá một cơ chế không được dữ liệu hỗ trợ, đồng thời chỉ ra thiết kế contour cụ thể cần được sửa hoặc từ bỏ trước khi dùng protected evaluation.

**Giải thích sâu:** Negative result có giá trị khi intervention, control, endpoint và decision rule đủ rõ để người đọc biết chính xác giả thuyết nào bị bác bỏ. Ở đây, kết luận chỉ áp dụng cho binary contour channel, matched Loop206 setup, 76 train-screen groups và ba selected seeds. Nó không chứng minh mọi boundary prior đều vô ích. Việc reject candidate trước protected test còn giảm nguy cơ protected-test fishing.

**Bằng chứng:** Slides 8, 9 và 17; lane Loop206 RQ2; artifact class: controlled ablation report và group-bootstrap summary.

**Bẫy cần tránh:** Nói contour luôn có hại hoặc biến một negative result có giới hạn thành định luật chung.

## B. Leakage, dataset, and split validity

### Q4 [P1] - “3 danh tính và 13 dòng cắt split” làm hỏng kết luận cũ như thế nào?
**Trả lời ngắn:** Audit legacy phát hiện 3 danh tính bệnh nhân tương ứng 13 dòng xuất hiện qua ranh giới split. Khi train và evaluation chứa thông tin tương quan cùng bệnh nhân, evaluation không còn độc lập theo bệnh nhân; metric vẫn tính đúng số học nhưng không còn hỗ trợ claim generalization độc lập.

**Giải thích sâu:** Vấn đề không phải công thức Dice bị sai mà là đơn vị độc lập bị vi phạm. Model có thể gặp trong train thông tin tương quan cùng bệnh nhân với evaluation, làm điểm số có nguy cơ lạc quan. Vì vậy, test-v2 legacy chỉ được giữ làm lịch sử có nhiễm, không dùng để xếp hạng khoa học. **Hỏi tiếp dễ gặp:** “Chỉ 3 identity thì ảnh hưởng có đáng kể không?” **Trả lời:** Độ lớn bias chưa được thiết lập bởi artifact hiện tại; chỉ cần crossing tồn tại là contract độc lập theo bệnh nhân đã bị vi phạm.

**Bằng chứng:** Slides 2 và 11; lane legacy contaminated evidence trước historical Clean-v3; artifact class: identity/split audit record.

**Bẫy cần tránh:** Tự ước lượng mức tăng metric do leakage hoặc nói toàn bộ 13 dòng đều giống hệt nhau.

### Q5 [P1] - Các số 2.008 train, 431 validation và 430 test nói lên điều gì, và không nói lên điều gì?
**Trả lời ngắn:** Đây là historical Clean-v3 metadata cho tổng 2.869 ảnh: 2.008 train, 431 adaptive development-validation và 430 sealed test. Chúng mô tả lane lịch sử đã ghi nhận; chúng không chứng minh prospective RQ1-v2 đã được admitted, vì verified index và integrity report của lane đó vẫn pending.

**Giải thích sâu:** Cần nói tên lane trước con số. Historical source-report audit ghi nhận split và không phát hiện cross-split identity-group overlap dưới các kiểm tra đã chạy. Prospective RQ1-v2 lại cần hash-bound index và zero-crossing integrity report riêng. **Hỏi tiếp dễ gặp:** “Có thể dùng ngay 2.008/431/430 làm dữ liệu xác nhận RQ1-v2 không?” **Trả lời:** Không. Đây là historical metadata; prospective RQ1-v2 admission chưa được thiết lập bởi artifact hiện tại.

**Bằng chứng:** Slide 5; lanes historical Paper RQ1/Clean-v3 và prospective RQ1-v2; artifact class: historical source-report audit plus blocked prospective protocol.

**Bẫy cần tránh:** Gọi 431 là test hoặc nói các con số lịch sử xác nhận dataset index của RQ1-v2.

### Q6 [P1] - Vì sao 431 ảnh validation không phải confirmatory evidence?
**Trả lời ngắn:** Partition 431 ảnh đã được dùng trong development, checkpoint hoặc model selection và promotion decisions. Khi quyết định được điều chỉnh theo cùng validation feedback, điểm cuối có selection optimism; vì vậy nó chỉ hỗ trợ descriptive point estimates, không hỗ trợ confirmatory superiority.

**Giải thích sâu:** Một tập confirmatory phải còn untouched sau khi hypothesis, model, preprocessing, threshold, stopping rule và analysis plan đã khóa. Ở đây, validation là adaptive development-validation, bất kể một nhãn cũ có thể gọi nó là protected. **Hỏi tiếp dễ gặp:** “Nếu không train trực tiếp trên 431 ảnh thì vẫn độc lập chứ?” **Trả lời:** Không đủ; việc dùng score để chọn checkpoint, model hoặc promotion cũng làm evaluation thích nghi với partition đó.

**Bằng chứng:** Slides 5, 7 và 13; lane historical Paper RQ1 adaptive development-validation; artifact class: selection/protocol record.

**Bẫy cần tránh:** Đồng nhất “không backpropagate trên validation” với “confirmatory untouched test”.

### Q7 [P2] - Có được nói Clean-v3 “không có leakage” không?
**Trả lời ngắn:** Không nên nói tuyệt đối. Cách đúng là historical Clean-v3 source-report audit “không phát hiện cross-split identity-group overlap” dưới các kiểm tra identity, exact duplicate, perceptual duplicate và split group đã ghi nhận. Audit giảm rủi ro đã kiểm tra, không chứng minh mọi quan hệ ẩn đều vắng mặt.

**Giải thích sâu:** Một audit chỉ có sức mạnh trong phạm vi identifier, similarity rule, threshold và nguồn metadata mà nó kiểm tra. Quan hệ sinh học chưa biết, nhãn sai, hoặc duplicate ngoài detector vẫn có thể tồn tại. Đồng thời không được lấy historical audit này thay cho prospective RQ1-v2 admission audit. Hai mệnh đề “historical audit không phát hiện crossing” và “prospective admission còn blocked” hoàn toàn có thể cùng đúng.

**Bằng chứng:** Slides 5 và 11; lanes historical Clean-v3 audit versus prospective RQ1-v2; artifact class: recorded leakage audit and pending integrity gate.

**Bẫy cần tránh:** Nói “đã chứng minh không leakage” hoặc phủ nhận historical audit chỉ vì RQ1-v2 chưa admitted.

### Q8 [P2] - Vì sao test-v3 và PH2 vẫn phải sealed?
**Trả lời ngắn:** Sealing bảo vệ hai partition khỏi adaptive selection. Candidate Loop206 đã fail train-screen gate, còn RQ1-v2 chưa hoàn tất verified admission và six-job protocol; mở test-v3 hoặc PH2 lúc này sẽ biến protected data thành thêm một development signal mà không tạo confirmatory evidence hợp lệ.

**Giải thích sâu:** Protected evaluation chỉ có ý nghĩa khi model, data index, preprocessing, metrics, thresholds, stopping rule và analysis đã khóa trước khi nhìn kết quả. Nếu mở partition để “xác nhận” kết quả mong muốn rồi tiếp tục sửa hệ thống, partition đó không còn untouched. Tài liệu hiện tại không báo cáo metric từ test-v3 hay PH2 và không dùng việc chưa mở chúng làm bằng chứng thành công hay thất bại.

**Bằng chứng:** Slides 5, 9, 10, 16 và 17; protected-test lane; artifact class: sealed-partition governance record.

**Bẫy cần tránh:** Nói sealed test đã xác nhận kết quả, hoặc đề nghị mở test chỉ để cứu một candidate đã fail gate.

## C. Baseline fairness and model-system differences

### Q9 [P1] - Vì sao nnU-Net là strong baseline nhưng không phải controlled architecture ablation?
**Trả lời ngắn:** nnU-Net v2 là strong complete-system baseline vì nó đại diện một pipeline segmentation mạnh và self-configuring. Nhưng L191 và L192 khác preprocessing, decoder, loss, augmentation, schedule, postprocessing, selection policy và geometry; vì vậy chênh lệch không thể quy riêng cho encoder hay architecture.

**Giải thích sâu:** Baseline mạnh trả lời “một hệ thống hoàn chỉnh cạnh tranh hoạt động ra sao”; controlled ablation trả lời “một thành phần gây ra thay đổi nào”. Paper RQ1 chỉ làm tốt câu hỏi thứ nhất trong contract cũ. **Hỏi tiếp dễ gặp:** “Nếu giữ nguyên dataset và metric thì có thể kết luận architecture tốt hơn không?” **Trả lời:** Không, vì nhiều biến hệ thống vẫn thay đổi đồng thời. Cần matched component controls hoặc one-factor ablation.

**Bằng chứng:** Slides 3, 6 và 12; lane historical Paper RQ1; artifact class: complete-system method comparison.

**Bẫy cần tránh:** Quy point-estimate direction riêng cho MiT-B3, decoder, raw RGB hoặc nnU-Net architecture.

### Q10 [P1] - Khác biệt 384x384 và 256x256 ảnh hưởng cách diễn giải ra sao?
**Trả lời ngắn:** IMP L191 chạy pipeline cố định 384x384; nnU-Net L192 dự đoán ở 256x256 rồi được khôi phục lên metric canvas 384x384. Resampling làm thay đổi localization và representation của biên, nên Dice và đặc biệt BF1 có thể đổi vì geometry, không chỉ vì model.

**Giải thích sâu:** Khi giảm rồi tăng resolution, vùng nhỏ và đường biên có thể bị lượng tử hóa hoặc làm mượt. Việc resize binary mask cũng không tương đương phục hồi probability trên original-image geometry rồi threshold. **Hỏi tiếp dễ gặp:** “Cùng chấm trên canvas 384 thì đã công bằng chưa?” **Trả lời:** Chưa; canvas cuối giống nhau không xóa khác biệt trong đường đi 256-to-384 và thông tin không gian đã mất trước đó.

**Bằng chứng:** Slides 6 và 12; lane historical Paper RQ1 older-geometry contract; artifact class: preprocessing/model geometry specification.

**Bẫy cần tránh:** Nói common 384 metric canvas tự động tạo fairness hoặc architecture là nguyên nhân duy nhất của BF1 khác biệt.

### Q11 [P2] - Một complete-system comparison công bằng hơn cần khóa những gì?
**Trả lời ngắn:** Cần cùng identities, original-image geometry, evaluation conditions, metric code, tuning budget, seeds và hardware accounting. Threshold, postprocessing và selection policy cũng phải được prespecify; nên có simple baseline, strong nnU-Net baseline và component ablations nếu muốn quy kết nguyên nhân.

**Giải thích sâu:** “Fair” không có nghĩa hai hệ thống phải giống nhau hoàn toàn; nó có nghĩa các lợi thế không liên quan đến câu hỏi phải được match hoặc công khai. Nếu preprocessing và training recipe là một phần của complete system, có thể giữ chúng, nhưng kết luận chỉ được ở system level. Muốn nói encoder hoặc contour gây thay đổi thì cần contrast chỉ thay đúng yếu tố đó.

**Bằng chứng:** Slides 6, 12 và 17; lanes historical Paper RQ1 và prospective RQ1-v2; artifact class: fairness limitations and registered next-experiment contract.

**Bẫy cần tránh:** Dùng chữ “fair” để che các khác biệt không được kiểm soát hoặc hứa rằng protocol tương lai chắc chắn đảo thứ hạng.

### Q12 [P2] - Vậy Paper RQ1 hiện hỗ trợ kết luận nào về IMP và nnU-Net?
**Trả lời ngắn:** Nó hỗ trợ một mô tả hẹp: trên historical adaptive development-validation và legacy geometry contract, complete nnU-Net ghi nhận point estimate cao hơn ở robust Dice, BF1 và Recall, còn IMP cao hơn ở Precision. Nó không hỗ trợ superiority, component causality hay protected-test generalization.

**Giải thích sâu:** Đây là một trade-off đa metric của hai systems, mỗi arm chỉ có một recorded run và không có paired confidence interval. Sự khác biệt quan sát được có thể đến từ nhiều thành phần của pipeline lẫn adaptive selection. Vì thế câu đúng là “ghi nhận point-estimate direction trong contract cũ”, không phải “nnU-Net tốt hơn IMP” theo nghĩa tổng quát.

**Bằng chứng:** Slides 6 và 7; lane historical Paper RQ1; artifact class: adaptive-validation point-estimate table.

**Bẫy cần tránh:** Chuyển một hướng point estimate thành ranking phổ quát hoặc claim statistically significant.

## D. Metrics, uncertainty, and statistical interpretation

### Q13 [P1] - Dice, IoU, BF1, Precision và Recall trả lời những khía cạnh khác nhau nào?
**Trả lời ngắn:** Dice và IoU đo overlap vùng; BF1 đo mức khớp đường biên trong tolerance contract; Precision phản ánh tỷ lệ pixel dự đoán lesion là đúng, còn Recall phản ánh tỷ lệ lesion reference được bắt lại. Vì chúng nhấn mạnh lỗi khác nhau, không nên dùng một metric duy nhất để tuyên bố hệ thống thắng.

**Giải thích sâu:** Dice và IoU có liên hệ đơn điệu nhưng scale khác nhau, nên phải giữ đúng metric contract thay vì hoán đổi số. BF1 nhạy với localization và geometry hơn overlap. Precision thấp có thể đi cùng over-segmentation; Recall thấp có thể đi cùng bỏ sót. **Hỏi tiếp dễ gặp:** “Có thể suy IoU từ Dice trên slide không?” **Trả lời:** Chỉ khi cùng aggregation và confusion contract được xác nhận; slide không hiển thị giá trị IoU cụ thể, nên không tự suy đoán hoặc công bố số.

**Bằng chứng:** Slides 3, 7 và 13; lane historical Paper RQ1 metric contract; artifact class: metric definitions and displayed point-estimate table.

**Bẫy cần tránh:** Đồng nhất Dice với IoU, gọi BF1 là overlap metric, hoặc bỏ qua Precision/Recall trade-off.

### Q14 [P1] - Vì sao chênh lệch 0.9019 so với 0.8959 chưa chứng minh superiority?
**Trả lời ngắn:** Đây là robust-Dice point estimates trên adaptive development-validation: nnU-Net 0.9019 và IMP 0.8959. Mỗi complete system chỉ có một recorded run, không có aligned paired per-case confidence interval hay hypothesis test; do đó chỉ được báo hướng quan sát, không được báo statistical superiority.

**Giải thích sâu:** Chênh lệch số học là 0.0060, nhưng độ lớn đó không tự cung cấp uncertainty. Adaptive selection còn làm estimate dễ lạc quan, và systems chạy dưới geometry khác nhau. **Hỏi tiếp dễ gặp:** “Số cao hơn thì ít nhất có thể nói chắc nnU-Net tốt hơn trên validation chứ?” **Trả lời:** Chỉ có thể nói point estimate được ghi nhận cao hơn trong contract đó; không thể mở rộng sang repeatability, population effect hoặc protected-test performance.

**Bằng chứng:** Slides 7 và 13; lane historical Paper RQ1; artifact class: single-run adaptive-validation results with no paired CI.

**Bẫy cần tránh:** Nói significant, superior, robustly better, hoặc coi 0.0060 là effect đã có uncertainty.

### Q15 [P2] - “Conditional on the three selected seeds” nghĩa là gì?
**Trả lời ngắn:** Loop206 average ba selected seeds và ba preregistered views trước khi bootstrap 10.000 lần theo 76 split groups. Interval vì vậy mô tả group variability khi ba seeds đó được giữ cố định; nó không ước lượng uncertainty của việc chọn seed hay quần thể training runs.

**Giải thích sâu:** Group bootstrap phù hợp hơn image bootstrap khi nhiều ảnh trong cùng group phụ thuộc nhau. Tuy nhiên, resample groups sau khi average seeds không tạo thêm seed replicates. Kết quả aggregate conditional on the three selected seeds là bằng chứng hiện có; hướng của từng seed chưa được thiết lập bởi bằng chứng paper được cấp quyền. Muốn có seed-population statement cần các independent runs được prespecify và một phân tích resample arm seeds lẫn split groups.

**Bằng chứng:** Slides 8, 9 và 13; lane Loop206 RQ2; artifact class: 10,000-resample split-group bootstrap summary.

**Bẫy cần tránh:** Gọi interval là seed-population CI hoặc nói cả ba seeds đều âm.

### Q16 [P2] - Vì sao prospective RQ1-v2 cần cả chín crossed contrasts?
**Trả lời ngắn:** Ba independent seeds cho mỗi arm tạo 3x3, tức chín, candidate-control contrasts. Báo cáo toàn bộ chín contrast giúp thấy arm-seed variability và tránh chỉ chọn same-seed hoặc cặp thuận lợi; execution pair trên cùng GPU là tổ chức vận hành, không tự tạo statistical pairing.

**Giải thích sâu:** Same numeric seed giữa hai hệ thống khác nhau không bảo đảm noise realizations có ý nghĩa paired. Phân tích tương lai cần tách independent-arm seed uncertainty khỏi split-group uncertainty, thay vì lấy ba cặp chạy theo lịch làm ba paired observations. Chín contrasts không tự giải quyết mọi vấn đề, nhưng khiến hướng kết quả theo tổ hợp seed minh bạch hơn và hỗ trợ seed-aware bootstrap đã đăng ký.

**Bằng chứng:** Slides 13 và 17; lane prospective RQ1-v2; artifact class: registered six-job/nine-contrast analysis plan, currently unexecuted.

**Bẫy cần tránh:** Nói chín contrasts đã tồn tại hoặc coi ba execution pairs là paired statistical samples.

## E. Loop206 ablation and negative results

### Q17 [P1] - Loop206 đã kiểm soát những biến nào để cô lập contour channel?
**Trả lời ngắn:** Control và candidate dùng cùng architecture, preprocessing, loss, budget, ba selected seeds và 20 epochs mỗi arm; cùng 308 fit groups, 76 train-screen holdout groups và ba preregistered views. Khác biệt được thiết kế là zero channel so với binary contour channel thứ tư.

**Giải thích sâu:** Đây là one-factor intervention mạnh hơn complete-system RQ1 vì phần lớn pipeline được match. Protected image hoặc mask không tham gia fitting, thresholding, selection hay evaluation. **Hỏi tiếp dễ gặp:** “Vậy causal claim đã hoàn toàn tổng quát chưa?” **Trả lời:** Không. Causal interpretation chỉ thuộc tested binary contour mechanism, dataset, train-screen partition, selected seeds và implementation hiện tại; external validity chưa được thiết lập.

**Bằng chứng:** Slide 8; lane Loop206 RQ2 matched train-screen; artifact class: locked ablation protocol.

**Bẫy cần tránh:** Gọi Loop206 là protected-test result hoặc mở rộng one-factor validity thành generalization ngoài protocol.

### Q18 [P1] - Kết quả -0.0313 Dice và interval nói chính xác điều gì?
**Trả lời ngắn:** Candidate trừ control cho robust Dice là -0.0313, với conditional 95% interval [-0.0491, -0.0156]. Trong matched 76-group train-screen protocol và conditional on the three selected seeds, aggregate effect nằm dưới zero nên contour candidate fail primary improvement gate; đây không phải kết luận cho từng seed.

**Giải thích sâu:** Dấu âm nghĩa là candidate có robust Dice thấp hơn control theo estimand đã khóa. Interval hoàn toàn dưới zero củng cố kết luận aggregate trong group-bootstrap contract, không phải seed-population certainty. BF1 delta là -0.0147 với interval [-0.0308, 0.0010], nên interval đó chạm qua zero. **Hỏi tiếp dễ gặp:** “Có thể nói contour chắc chắn làm hại mọi run không?” **Trả lời:** Không; bằng chứng hiện có chỉ thiết lập kết quả aggregate conditional on the three selected seeds. Hướng của từng seed chưa được paper authority thiết lập, và seed-selection uncertainty chưa được ước lượng.

**Bằng chứng:** Slide 9; lane Loop206 RQ2; artifact class: conditional group-bootstrap result table.

**Bẫy cần tránh:** Đổi candidate-minus-control thành control-minus-candidate, bỏ chữ conditional, hoặc nói mọi seed đều âm.

### Q19 [P1] - Negative result này bác bỏ điều gì và không bác bỏ điều gì?
**Trả lời ngắn:** Nó bác bỏ việc promote binary contour channel đã kiểm tra dưới primary matched Loop206 train-screen rule. Nó không bác bỏ mọi contour representation, mọi boundary-aware method, mọi dataset, hay khả năng một thiết kế khác hoạt động; những mệnh đề rộng hơn cần thí nghiệm mới.

**Giải thích sâu:** Một negative ablation có scope theo intervention, control, data, training recipe, metric và decision rule. Cơ chế saliency/contour khác, loss khác hoặc original-image protocol khác là hypothesis mới chứ không phải ngoại suy của Loop206. **Hỏi tiếp dễ gặp:** “Có nên thử tiếp cho đến khi contour dương không?” **Trả lời:** Chỉ khi có mechanism hypothesis và protocol mới được prespecify; không được dùng protected partitions để lặp tuning đến khi có kết quả thuận lợi.

**Bằng chứng:** Slides 9 và 17; lane Loop206 RQ2; artifact class: negative-ablation decision record.

**Bẫy cần tránh:** Nói “contour vô dụng” hoặc dùng test-v3/PH2 như development set để cứu giả thuyết.

### Q20 [P2] - Có thể kiểm toán mọi Loop206 gate failure từ bảng hiện tại không?
**Trả lời ngắn:** Chưa. Paper hiển thị aggregate Dice và boundary F1 deltas cùng intervals, nhưng đánh dấu thresholds, per-seed directions và các endpoint chưa báo cáo là `unavailable`; từng gate status vẫn `blocked`. Không được tự suy ra endpoint chưa công bố từ slide hoặc prose cũ.

**Giải thích sâu:** Prose có thể ghi candidate fail nhiều gates, nhưng independent audit cần một endpoint table để tái dựng từng verdict. Bảng bổ sung phù hợp phải chứa gate name, threshold, observed aggregate, interval nếu có, pass/fail và cả ba seed directions. Cho đến khi artifact đó tồn tại, câu trả lời chỉ nên dựa vào Dice và BF1 đã hiển thị, cùng bounded overall rejection.

**Bằng chứng:** Slides 9 và 13; lane Loop206 RQ2; artifact class: displayed aggregate table plus documented endpoint-audit limitation.

**Bẫy cần tránh:** Bịa số Precision, Recall, HD95, ASSD hoặc nói mọi gate value đã independently auditable.

## F. Demo, runtime identity, and failure handling

### Q21 [P1] - Paper RQ1, fixed L206 cache và live demo khác nhau ở đâu?
**Trả lời ngắn:** Paper RQ1 là historical L191/L192 adaptive-validation comparison. Fixed L206 cache là audited illustrative lane so sánh zero channel với contour channel trên mẫu train-screen cố định. Live lane gửi cùng RGB qua `L206-control-s206` rồi reconstructed `L192-nnUNet-v2-raw-100ep`; live output không phải Paper RQ1 hay fixed-cache evidence.

**Giải thích sâu:** Ba lane trả lời ba câu hỏi khác nhau: historical score direction, fixed qualitative disagreement của matched ablation, và behavior của current runtime trên cùng input. Prospective RQ1-v2 là lane thứ tư liên quan ở đây nhưng vẫn pending. **Hỏi tiếp dễ gặp:** “Nếu model ID giống L192 thì live mask có thể minh họa Paper RQ1 không?” **Trả lời:** Không. Model ID/checkpoint binding không thiết lập original-runtime equivalence, historical metric contract hoặc ground-truth evaluation.

**Bằng chứng:** Slides 4, 10, 14 và 16; lanes Paper RQ1, fixed L206 cache, live reconstructed lane và RQ1-v2; artifact class: release manifest plus approved demo script.

**Bẫy cần tránh:** Dùng từ “L206 demo” mà không nói fixed hay live, hoặc đưa live mask vào Paper RQ1 claim.

### Q22 [P1] - Hai public demo samples có độc lập với training không?
**Trả lời ngắn:** Release manifest ghi nhận cả hai samples có trong 2.008 training rows của L192, đồng thời nằm ngoài 308 fit groups của L206 control và thuộc 76-group train-screen holdout. Vì exposure bất đối xứng, chúng chỉ minh họa workflow; không phải comparison về generalization.

**Giải thích sâu:** Câu trả lời phải tách observed manifest metadata khỏi independent provenance. Exposure của L192 và exclusion khỏi L206 fit đã được ghi nhận trong manifest; provenance độc lập vượt ngoài manifest chưa được xác minh. **Hỏi tiếp dễ gặp:** “Nếu L206 chưa fit các samples thì có thể nói L206 generalize tốt hơn không?” **Trả lời:** Không, vì L192 đã exposure, không có loaded GT, và sample selection không tạo một fair held-out comparison.

**Bằng chứng:** Slide 10; live reconstructed lane; artifact class: observed release-manifest training-exposure metadata, independent provenance unverified.

**Bẫy cần tránh:** Nói cả hai samples đều unseen, hoặc biến exposure bất đối xứng thành lợi thế accuracy cho một arm.

### Q23 [P1] - Vì sao demo không có ground truth thì không được nói accuracy?
**Trả lời ngắn:** Live lane có trạng thái `ground_truth_not_loaded`; ba panel chỉ là original, IMP và reconstructed nnU-Net. Không có reference mask thì không tính được Dice, IoU, BF1, Precision hay Recall, nên receipt và mask nhìn đẹp chỉ chứng minh workflow hiển thị, không chứng minh accuracy.

**Giải thích sâu:** Qualitative plausibility dễ đánh lừa vì người xem có thể thích một contour mà không biết lesion boundary thật. Receipt bind transaction identity và latency, không phải semantic correctness. **Hỏi tiếp dễ gặp:** “Hai model cho mask gần nhau có phải là corroboration không?” **Trả lời:** Không; hai model có thể cùng sai, và agreement không thay thế authorized representative ground truth cùng metric protocol.

**Bằng chứng:** Slides 10 và 14; live illustrative lane; artifact class: demo transaction record with `ground_truth_not_loaded`.

**Bẫy cần tránh:** Gọi output đẹp là chính xác, generalizable, fair hoặc có giá trị chẩn đoán.

### Q24 [P2] - Reconstructed nnU-Net hiện chứng minh được identity nào?
**Trả lời ngắn:** Startup manifest và checks nhận diện cấu hình runtime cùng model identity hiện hành. Hashes chỉ bind exact bytes của checkpoint hoặc artifact được liệt kê; chúng không bind toàn bộ runtime hay chứng minh tương đương với historical original L192 execution environment. Evidence hiện tại còn ghi nhận same-input mask-byte repeatability của reconstructed runtime bị blocked.

**Giải thích sâu:** Runtime equivalence cần nhiều hơn một checkpoint hash: preprocessing code, library/runtime versions, plans, postprocessing, deterministic controls và output behavior đều phải được đối chiếu. Observed drift thuộc các reconstructed configurations đã thử; nguyên nhân vẫn speculative và không được suy rộng sang original runtime không còn sẵn để kiểm tra.

**Bằng chứng:** Slides 10, 14 và 16; live reconstructed lane; artifact class: current runtime identity manifests and failed A/B/A repeatability records.

**Bẫy cần tránh:** Nói hash giống nhau chứng minh complete runtime equivalence hoặc bịa nguyên nhân nondeterminism.

### Q25 [P2] - Receipt và fail-closed behavior bảo vệ demo như thế nào?
**Trả lời ngắn:** Complete receipt chỉ được phát sau khi cả IMP và reconstructed nnU-Net hoàn thành trên cùng input binding. Nếu một arm lỗi, hệ thống giữ output IMP hiện tại nhưng xóa panel nnU-Net và receipt để không tái dùng stale evidence; hiện release vẫn chưa có canonical live receipt.

**Giải thích sâu:** Fail-closed nghĩa là thiếu prerequisite hoặc mismatch phải làm claim dừng rõ ràng, không âm thầm dùng cache hay output cũ. Presenter S đã quan sát oversize failure clearing và recovery qua callback, nhưng browser state và cleanup acceptance chưa pass. Receipt nếu có chỉ bind transaction identity/latency; nó không chứng minh determinism, accuracy, privacy hay deploy-readiness.

**Bằng chứng:** Slides 10 và 14; live illustrative lane; artifact class: Presenter S callback transcript and blocked canonical acceptance packet.

**Bẫy cần tránh:** Nói một HTTP 200 hoặc closed ports chứng minh complete E2E acceptance.

### Q26 [P2] - Audit định tính OOD cho thấy failure mode nào, và được phép kết luận đến đâu?
**Trả lời ngắn:** Bản tóm tắt local qualitative OOD audit ghi nhận zero paired qualitative passes và reconstructed nnU-Net có systematic background flooding trên các trường hợp đã xem. Đây chỉ là observed local qualitative failure pattern; nó không phải accuracy estimate, fairness audit hay population-level OOD benchmark.

**Giải thích sâu:** “Zero paired passes” nghĩa là không cặp output nào trong audit cục bộ thỏa tiêu chí định tính ghép cặp đã dùng; “background flooding” mô tả mask nnU-Net tràn có hệ thống vào nền trong các mẫu được review. Không có representative sampling, authorized paired ground truth hoặc quantitative denominator đủ để suy ra tỷ lệ lỗi. Kết quả này biện minh cho việc không dùng random uploads để quảng bá chất lượng và cho fail-closed/static fallback, nhưng không định lượng generalization hay subgroup fairness.

**Bằng chứng:** Slide 10; live reconstructed/OOD lane; artifact class: sanitized local qualitative OOD review summary, source locator intentionally omitted.

**Bẫy cần tránh:** Biến audit định tính cục bộ thành accuracy, fairness, clinical safety hoặc kết luận rằng mọi OOD input đều flood.

## G. Reproducibility, deployment, ethics, and limitations

### Q27 [P1] - Trạng thái hiện tại của prospective RQ1-v2 là gì?
**Trả lời ngắn:** RQ1-v2 vẫn pending/unverified. Protocol có kế hoạch hai arms nhân ba seeds, nhưng verified dataset index, integrity admission, sáu locked configs, sáu validated job receipts, chín crossed contrasts và digest-bound report chưa hoàn tất; không số mới nào được promote vào paper hoặc deck.

**Giải thích sâu:** Seed declarations không phải completed runs. Lane prospective chỉ được mở sau khi hash-bound index pass zero-crossing audit, model/runtime/input manifests được khóa, sáu jobs hoàn thành và analysis đúng registered contract. **Hỏi tiếp dễ gặp:** “Có thể dùng historical 2.008/431 hoặc fixed cache thay receipt còn thiếu không?” **Trả lời:** Không. Đó là evidence lanes khác; substitution sẽ phá prospective admission contract. Prospective RQ1-v2 admission chưa được thiết lập bởi artifact hiện tại.

**Bằng chứng:** Slides 5, 16 và 17; lane prospective RQ1-v2; artifact class: blocked protocol scaffold and missing six-job evidence.

**Bẫy cần tránh:** Đếm ba seed IDs thành sáu completed jobs hoặc nói RQ1-v2 đã admitted.

### Q28 [P2] - Một lab khác có thể tái lập điều gì từ repository hiện tại?
**Trả lời ngắn:** Lab khác có thể audit tracked source contracts, manifests, hashes và available audited reports. Họ chưa thể rerun đầy đủ historical L191/L192/Loop206 experiments hoặc prospective six-job study từ fresh clone vì còn thiếu external data, weights, caches, configs, paired predictions và một số historical reports.

**Giải thích sâu:** Digest-recorded giúp xác nhận bytes nào đã được tham chiếu; strict local audit xác nhận source bytes trong lần audit cụ thể. Reproduction mạnh hơn đòi data acquisition/checksums, executable training code, exact configs, environment locks, checkpoints, seed schedule, compute requirements và expected outputs. Current reconstructed live service cũng không thay thế historical original runtime.

**Bằng chứng:** Slides 15 và 16; historical Paper RQ1/reproducibility lane; artifact class: manifests and historical reproducibility audit.

**Bẫy cần tránh:** Gọi repository clone-runnable cho toàn bộ experiment hoặc đồng nhất audit hash với rerun độc lập.

### Q29 [P2] - Hash, live display và package hiện tại có chứng minh deploy-readiness hay clinical validity không?
**Trả lời ngắn:** Không. Hash chỉ bind exact bytes của checkpoint hoặc artifact được liệt kê; startup manifest và checks nhận diện cấu hình runtime cùng model identity hiện hành. Live display cho thấy một workflow quan sát được; package hỗ trợ trình bày bounded claims. Không cơ chế nào tự chứng minh accuracy, fairness, privacy compliance, deterministic runtime, monitoring, clinical utility, safety hay capacity cần cho deployment.

**Giải thích sâu:** Deploy-readiness cần validated performance trên representative locked data, robustness/error slices, calibration khi phù hợp, security/privacy controls, deterministic or bounded runtime behavior, monitoring, rollback và operational acceptance. Clinical validity còn cần target population, intended use, qualified reference standard và governance phù hợp. Hiện browser/runtime acceptance bị blocked, demo không có GT, và test-v3/PH2 vẫn sealed.

**Bằng chứng:** Slides 1, 10, 14, 16 và 17; live/release/clinical-boundary lanes; artifact class: release manifests, blocked runtime acceptance and ethics limitations.

**Bẫy cần tránh:** Dùng “reproducible deployment” như đồng nghĩa với an toàn lâm sàng hoặc coi matching hash là quality measurement.

### Q30 [P2] - Thí nghiệm tiếp theo nào có thể hỗ trợ kết luận mạnh hơn?
**Trả lời ngắn:** Sau verified RQ1-v2 admission, chạy đúng ba independent seeds cho mỗi complete-system arm trên cùng original-image geometry, conditions và metric code; validate sáu receipts, báo cáo cả chín crossed contrasts, rồi bootstrap độc lập theo arm seeds và split groups trước khi cân nhắc protected test.

**Giải thích sâu:** Kế hoạch cần prespecify simple và strong baselines, tuning budget, threshold/postprocessing, multiplicity rule và error slices theo source, size, quality, corruption và tails nếu contract yêu cầu. Kết quả phải được digest-bind trước evidence promotion. Đây là registered next step chứ không phải completed evidence; cũng chưa đủ cho clinical hoặc fairness claim nếu chưa có representative subgroup design và reference standard tương ứng.

**Bằng chứng:** Slide 17; lane prospective RQ1-v2; artifact class: registered next-experiment plan, currently blocked/unexecuted.

**Bẫy cần tránh:** Cam kết future result, mở protected test trước khi khóa protocol, hoặc nói six-job design tự động chứng minh fairness/clinical readiness.
