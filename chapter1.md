%=================================================
\chapter{Giới thiệu}
\label{chap:introduction}
%=================================================

\section{Đặt vấn đề}
\label{sec:dat_van_de}

Xu hướng già hóa dân số toàn cầu đang diễn ra với tốc độ chưa từng có. Theo Tổ chức Y tế Thế giới (WHO), trên toàn thế giới mỗi năm có hơn 684.000 ca tử vong do té ngã, khiến té ngã trở thành nguyên nhân gây tử vong do tai nạn thương tích vô ý đứng thứ hai toàn cầu, tập trung chủ yếu ở nhóm người từ 60 tuổi trở lên. Tại Việt Nam, báo cáo của Bộ Y tế cho thấy tai nạn té ngã ở người cao tuổi chiếm khoảng 30\% tổng số tai nạn thương tích, gây gánh nặng đáng kể cho cả hệ thống y tế và kinh tế -- xã hội. Ngoài các chấn thương thể xác như gãy xương và chấn thương sọ não, té ngã còn dẫn đến ``hội chứng sợ té ngã'' (\textit{fear of falling}), gây giảm vận động, cô lập xã hội và trầm cảm -- tạo vòng xoắn tiêu cực làm tăng nguy cơ té ngã lần tiếp theo.

Xuất phát từ thực tiễn đó, nhu cầu xây dựng hệ thống phát hiện té ngã theo thời gian thực trong môi trường giám sát thông minh trở nên cấp thiết. Tuy nhiên, các phương pháp hiện tại vẫn tồn tại hạn chế đáng kể: cảm biến đeo (\textit{wearable sensors}) phụ thuộc vào việc người dùng mang thiết bị liên tục, dẫn đến tỷ lệ tuân thủ thấp (\textit{poor user compliance}) do sự bất tiện và giới hạn về pin; xử lý ảnh RGB trực tiếp xâm phạm quyền riêng tư, nhạy cảm với điều kiện ánh sáng và che khuất; mô hình CNN/LSTM cổ điển có chi phí tính toán cao, khó học phụ thuộc dài hạn, và thiếu cơ chế attention toàn cục. Những hạn chế này tạo ra khoảng trống khoa học cần được giải quyết.

\section{Động lực và mục tiêu nghiên cứu}
\label{sec:dong_luc_va_muc_tieu}

\subsection{Động lực nghiên cứu}
\label{sec:dong_luc_nghien_cuu}

Sự phát triển mạnh mẽ của các kỹ thuật ước lượng tư thế (\textit{Pose Estimation}) trong thị giác máy tính đã mở ra một hướng tiếp cận hoàn toàn mới: phát hiện té ngã dựa trên khung xương (\textit{skeleton-based fall detection}) sử dụng trí tuệ nhân tạo lai. Thay vì xử lý ảnh RGB thô hoặc dữ liệu cảm biến quán tính, hệ thống sử dụng mô hình Pose Estimation để trích xuất 17 điểm khóa COCO bao gồm tọa độ (x, y) và độ tin cậy (\textit{confidence}) của các bộ phận cơ thể, vừa đảm bảo độ chính xác cao vừa bảo vệ quyền riêng tư người dùng.

Kiến trúc YOLOv11-Pose tích hợp các khối C3k2 và C2PSA mang lại khả năng trích xuất khung xương hiệu quả về mặt tính toán, bất biến theo tỷ lệ với khoảng cách camera, và độ chính xác cao trên tập dữ liệu chuẩn COCO. Việc sử dụng đặc trưng khung xương thay cho ảnh thô đồng thời giải quyết triệt để vấn đề xâm phạm quyền riêng tư. Tiếp theo, kiến trúc Hybrid Transformer mô hình hóa động học thời gian nhờ cơ chế Self-Attention -- cho phép học mối quan hệ phụ thuộc giữa các khung hình ở mọi khoảng cách thời gian một cách song song, khắc phục nhược điểm vanishing gradient của LSTM. Đồng thời, toán tử Mean Pooling thay vì các lớp Transformer đầy đủ giúp giảm đáng kể chi phí tính toán, tránh chi phí bậc hai (\textit{quadratic cost}) của attention nguyên bản, phù hợp cho triển khai Edge AI.

Điểm khác biệt cốt lõi so với các phương pháp trước đó: (1) hệ thống không lưu trữ hay xử lý ảnh RGB thô, chỉ hoạt động trên vector tọa độ khung xương 17 điểm; (2) kiến trúc lai YOLOv11-Pose + Hybrid Transformer nhẹ hơn đáng kể so với 3D-CNN truyền thống nhưng vẫn đảm bảo khả năng biểu diễn động học phức tạp; (3) thiết kế tối ưu cho triển khai Edge AI với tốc độ suy luận đáp ứng yêu cầu thời gian thực.

\subsection{Mục tiêu nghiên cứu}
\label{sec:muc_tieu_nghien_cuu}

\textbf{Mục tiêu tổng quát}: Thiết kế và xây dựng một hệ thống phát hiện té ngã cho người cao tuổi theo thời gian thực sử dụng kiến trúc lai giữa YOLOv11-Pose và Hybrid Transformer, có khả năng hoạt động ổn định trong môi trường giám sát trong nhà, bảo vệ quyền riêng tư người dùng, và triển khai được trên các thiết bị edge với tài nguyên phần cứng hạn chế.

\textbf{Mục tiêu cụ thể}:

\begin{itemize}
    \item Nghiên cứu, so sánh và lựa chọn kiến trúc Pose Estimation tối ưu (YOLOv11-Pose, YOLOv8-Pose) trên tập dữ liệu MCFD và CaucaFall.
    
    \item Xây dựng vector đặc trưng PIFR 60 chiều: 51 giá trị từ 17 điểm khóa COCO (tọa độ x, y và độ tin cậy) cùng 9 góc hình học, với hàm tính góc arccos sử dụng $\epsilon = 10^{-8}$ chống lỗi ZeroDivisionError.
    
    \item Xây dựng mô hình lai Hybrid Transformer (Self-Attention + Mean Pooling) thay thế LSTM/TCN truyền thống để mô hình hóa động học thời gian.
    
    \item Xây dựng pipeline tiền xử lý tạo tập dữ liệu AIO (\textit{All-In-One}) từ CaucaFall và MCFD, chuẩn hóa shape (60, 60) cho mô hình Transformer.
    
    \item Triển khai hệ thống thời gian thực với giao diện PyQt5, cơ chế sliding window (stride=15, maxlen=60), và module cảnh báo Telegram với cooldown 10 giây.
    
    \item Áp dụng chiến lược Data Augmentation Isolation tránh data leakage và kỹ thuật quản lý bộ nhớ (cap.release(), del, gc.collect()) đảm bảo huấn luyện ổn định.
\end{itemize}

\section{Đối tượng và phạm vi nghiên cứu}
\label{sec:doi_tuong_va_pham_vi}

\subsection{Đối tượng nghiên cứu}
\label{sec:doi_tuong_nghien_cuu}

Đối tượng nghiên cứu của luận văn bao gồm:

\begin{itemize}
    \item \textbf{Mô hình YOLOv11-Pose}: Nghiên cứu kiến trúc, cơ chế hoạt động, các khối C3k2 và C2PSA, phương pháp trích xuất 17 điểm khóa COCO, và chiến lược huấn luyện lại (\textit{fine-tuning}).
    
    \item \textbf{Kiến trúc Transformer trong thị giác máy tính}: Nghiên cứu cơ chế Self-Attention, các biến thể (Vanilla, Hybrid), Mean Pooling, và ứng dụng trong mô hình hóa chuỗi thời gian.
    
    \item \textbf{Bài toán phân loại nhị phân Fall/Non-fall}: Nghiên cứu đặc điểm của hành động té ngã so với các hoạt động sinh hoạt hàng ngày (\textit{Activities of Daily Living -- ADL}) như đi, ngồi, đứng, nằm, cúi người.
    
    \item \textbf{Tập dữ liệu}: Tập dữ liệu CaucaFall và MCFD (\textit{Multiple Cameras Fall Dataset}) làm cơ sở huấn luyện và đánh giá.
\end{itemize}

\subsection{Phạm vi nghiên cứu}
\label{sec:pham_vi_nghien_cuu}

Luận văn giới hạn phạm vi nghiên cứu trong các ràng buộc sau:

\begin{itemize}
    \item \textbf{Bài toán phân loại nhị phân}: Hệ thống phân loại đầu ra là Fall (1) / Non-fall (0) -- nhận biết hai lớp: té ngã và không té ngã. Không mở rộng sang phân loại đa lớp (loại té ngã: ngã ra sau, ngã về phía trước, trượt chân, v.v.).
    
    \item \textbf{Môi trường giám sát trong nhà}: Hệ thống được thiết kế và đánh giá trong môi trường camera giám sát trong nhà (\textit{indoor surveillance}) với một đến nhiều camera, góc nhìn ngang tầm hoặc trên cao. Không thử nghiệm trong môi trường ngoài trời với các yếu tố thời tiết, ánh sáng phức tạp.
    
    \item \textbf{Triển khai Edge AI}: Mô hình được thiết kế tối ưu cho triển khai trên thiết bị edge (GPU nhúng NVIDIA Jetson, hoặc CPU hiệu năng cao), với ràng buộc tốc độ suy luận từ 25 FPS trở lên và mức tiêu thụ bộ nhớ từ 4GB VRAM trở xuống.
    
    \item \textbf{Không bao gồm}: Hệ thống không xử lý đa đối tượng trong cùng một khung hình (\textit{multi-person tracking}), không bao gồm module re-identification, và không tích hợp nhận dạng khuôn mặt hay các thông tin nhận dạng cá nhân khác.
\end{itemize}

\section{Phương pháp và kiến trúc đề xuất}
\label{sec:phuong_phap_va_kien_truc}

\subsection{Kiến trúc tổng thể hệ thống}
\label{sec:kien_truc_tong_the}

Hệ thống phát hiện té ngã đề xuất được xây dựng theo kiến trúc pipeline xử lý theo thời gian thực, gồm 4 giai đoạn chính. \textit{Giai đoạn 1 (Pose Estimation)} sử dụng YOLOv11-Pose để trích xuất 17 điểm khóa COCO từ mỗi khung hình của video đầu vào. \textit{Giai đoạn 2 (Feature Extraction)} tính toán vector đặc trưng PIFR 60 chiều từ các điểm khóa đã trích xuất. \textit{Giai đoạn 3 (Temporal Modeling)} sử dụng Hybrid Transformer để mô hình hóa động học thời gian từ chuỗi vector đặc trưng. \textit{Giai đoạn 4 (Classification)} phân loại đầu ra là Fall hoặc Non-fall thông qua lớp phân loại nhị phân.

Hệ thống sử dụng cơ chế sliding window với stride=15 và maxlen=60 để duy trì bộ đệm 60 khung hình liên tục, đảm bảo đầu vào mô hình luôn có kích thước cố định (60, 60) bất kể độ dài video gốc. Khi phát hiện té ngã, hệ thống kích hoạt module cảnh báo Telegram với cooldown 10 giây.

\subsection{Mô hình YOLOv11-Pose}
\label{sec:mo_hinh_yolo_pose}

YOLOv11-Pose là phiên bản Pose Estimation của mô hình YOLOv11, sử dụng kiến trúc CSP (\textit{Cross Stage Partial}) với các khối C3k2 và C2PSA. Khối C3k2 (\textit{CSP Bottleneck with 3 konvolutions}) thay thế các khối Bottleneck truyền thống bằng ba lớp tích chập song song, tăng cường khả năng trích xuất đặc trưng đa масштаб mà không tăng đáng kể chi phí tính toán. Khối C2PSA (\textit{Convolutional Pyramid Spatial Attention}) tích hợp cơ chế attention không gian dạng kim tự tháp, cho phép mô hình tập trung vào các vùng quan trọng của cơ thể người trong khung hình.

Mô hình trích xuất 17 điểm khóa COCO: Nose(0), L-Shoulder(5), R-Shoulder(6), L-Hip(11), R-Hip(12), L-Knee(13), R-Knee(14), L-Ankle(15), R-Ankle(16), cùng 8 điểm phụ trợ khác. Mỗi điểm khóa được biểu diễn bởi bộ ba giá trị (nx, ny, conf), trong đó nx, ny là tọa độ đã chuẩn hóa theo chiều rộng và chiều cao khung hình tương ứng, conf là độ tin cậy của dự đoán.

Ưu điểm nổi bật của YOLOv11-Pose so với các Pose Estimation model khác: (1) bất biến theo tỷ lệ (\textit{scale-invariant}) -- hoạt động tốt với người ở mọi khoảng cách camera; (2) tốc độ suy luận cao, phù hợp cho xử lý thời gian thực; (3) chỉ xuất ra tọa độ khung xương, không lưu trữ ảnh thô -- bảo vệ quyền riêng tư người dùng; (4) khả năng hoạt động ổn định trong điều kiện ánh sáng yếu và che khuất một phần.

Khi không phát hiện được người trong khung hình, hệ thống sử dụng chiến lược zero-fallback: nhân bản vector 60 chiều từ khung hình trước đó (hoặc vector zero nếu là khung hình đầu tiên) nhằm duy trì tính toàn vẹn chuỗi thời gian.

\subsection{�ặc trưng PIFR}
\label{sec:dac_trung_pifr}

PIFR (\textit{Pose-Informed Fall Recognition}) là hệ thống trích xuất 9 đặc trưng hình học phức tạp từ 17 điểm khóa COCO đã chuẩn hóa, mang lại khả năng biểu diễn ngữ nghĩa vượt trội so với các vector đặc trưng thô. 9 đặc trưng PIFR bao gồm:

\begin{itemize}
    \item \textbf{F1, F2} (Center of Mass): Trọng tâm X và Y của cơ thể -- cho biết vị trí tổng thể của người trong khung hình.
    \item \textbf{F3} (Shoulder-Nose Angle): Góc vai-mũi -- phản ánh độ nghiêng đầu và thân trên.
    \item \textbf{F4} (Torso Angle): Góc thân từ mũi đến trung điểm hông, chỉ báo quan trọng nhất để phân biệt té ngã với ngồi xuống.
    \item \textbf{F5} (Hip Angle): Góc hông -- độ mở rộng của đường hông, phân biệt tư thế đứng với ngồi.
    \item \textbf{F6} (Shoulder Angle): Góc vai -- biên độ chuyển động ngang của vai.
    \item \textbf{F7, F8} (Left/Right Leg Angles): Góc chân trái và chân phải -- phát hiện dáng đi loạng choạng và mất thăng bằng.
    \item \textbf{F9} (Nose-to-Ankle Angle): Góc mũi-mắt cá chân -- chỉ báo cuối cùng về sự ngã gần như thẳng đứng.
\end{itemize}

Tất cả các góc được tính bằng công thức hình học vector với hàm arccos:

$$\theta = \arccos\left(\text{np.clip}\left(\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\| + \epsilon}, -1.0, 1.0\right)\right)$$

với tham số an toàn $\epsilon = 10^{-8}$ chống lỗi chia cho zero (\textit{ZeroDivisionError}) khi hai vector có độ dài bằng 0 do các điểm khóa bị che khuất hoặc trùng nhau. Toán tử \texttt{np.clip} đảm bảo đầu vào của \texttt{arccos} luôn nằm trong đoạn $[-1.0, 1.0]$, tránh giá trị NaN.

Vector đặc trưng PIFR 60 chiều cuối cùng được ghép nối: 51 giá trị từ 17 COCO keypoints (nx, ny, conf) $\times$ 17 + 9 góc hình học = \textbf{60 dimensions}.

\subsection{Mạng Hybrid Transformer}
\label{sec:mang_hybrid_transformer}

Kiến trúc Hybrid Transformer được thiết kế để mô hình hóa động học thời gian từ chuỗi vector đặc trưng PIFR 60 chiều. Đầu vào là ma trận $(T, 60)$ với $T = 60$ khung hình từ cơ chế sliding window. Đầu ra là xác suất phân loại Fall/Non-fall.

Kiến trúc gồm hai thành phần chính. \textit{Self-Attention layer}: sử dụng cơ chế attention đa đầu (\textit{multi-head attention}) cho phép mô hình học được mối quan hệ phụ thuộc giữa các khung hình ở mọi khoảng cách thời gian một cách song song, khắc phục hoàn toàn nhược điểm vanishing gradient của LSTM. Cơ chế attention được tính theo công thức:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

trong đó $Q$ (Query), $K$ (Key), $V$ (Value) được tạo từ ma trận đầu vào thông qua các phép biến đổi tuyến tính, và $d_k$ là chiều của vector Key.

\textit{Mean Pooling}: sau lớp Self-Attention, toán tử Mean Pooling tính trung bình theo chiều thời gian để tổng hợp thông tin từ toàn bộ chuỗi 60 khung hình thành một vector biểu diễn duy nhất. Mean Pooling thay thế cho các lớp Transformer đầy đủ (\textit{full Transformer layers}) giúp giảm đáng kể chi phí tính toán, tránh chi phí bậc hai $O(T^2)$ của attention nguyên bản, phù hợp với triển khai Edge AI.

Việc sử dụng Hybrid Transformer thay thế hoàn toàn LSTM và TCN mang lại ba ưu điểm cốt lõi: (1) khả năng học phụ thuộc dài hạn mà không cần nhiều bước tuần tự; (2) tính toán song song hoàn toàn, tăng tốc độ huấn luyện và suy luận; (3) cơ chế attention tự động tập trung vào các khung hình quan trọng nhất (thời điểm mất thăng bằng, ngã) mà không cần thiết kế thủ công.

\subsection{Hệ thống cảnh báo thời gian thực}
\label{sec:he_thong_canh_bao}

Hệ thống cảnh báo thời gian thực bao gồm ba thành phần: cơ chế sliding window, module phát hiện té ngã, và module gửi cảnh báo Telegram.

\textbf{Sliding Window}: Hệ thống sử dụng deque data structure với tham số \texttt{stride=15} và \texttt{maxlen=60}. Mỗi khung hình mới được thêm vào cuối deque, và deque tự động loại bỏ khung hình cũ nhất khi đạt đến giới hạn 60 phần tử. Với stride=15, deque mới được đưa vào mô hình phân loại mỗi khi có đủ 60 khung hình -- tức tần suất kiểm tra là mỗi 15 khung hình mới. Thiết kế này đảm bảo đầu vào mô hình luôn có kích thước cố định (60, 60) và giảm tải tính toán so với xử lý mọi khung hình.

\textbf{Module Telegram IoT}: Khi mô hình phân loại đầu ra là Fall (xác suất > ngưỡng), hệ thống gửi tin nhắn cảnh báo qua Telegram Bot API. Tin nhắn bao gồm nhãn ``CẢNH BÁO: Phát hiện té ngã!'' kèm thời gian và hình ảnh khung hình sự cố. Cơ chế \textbf{cooldown 10 giây} (\textit{10-second cooldown mechanism}) đảm bảo rằng sau khi một cảnh báo được gửi, hệ thống sẽ không gửi cảnh báo mới trong vòng 10 giây tiếp theo, ngay cả khi mô hình tiếp tục phát hiện té ngã. Điều này ngăn chặn hiện tượng spam thông báo gây phiền toái cho người nhận.

\section{Đóng góp của đề tài}
\label{sec:dong_gop_cua_de_tai}

Luận văn đóng góp các kết quả chính sau:

\begin{enumerate}
    \item \textbf{Về mô hình}: Đề xuất kiến trúc lai YOLOv11-Pose + Hybrid Transformer, kết hợp ưu điểm của Pose Estimation bất biến theo tỷ lệ với khả năng mô hình hóa động học thời gian của Transformer. Kiến trúc nhẹ hơn đáng kể so với 3D-CNN truyền thống, phù hợp với ràng buộc tài nguyên của thiết bị edge.
    
    \item \textbf{Về đặc trưng}: Xây dựng và công thức hóa hệ thống trích xuất PIFR với 9 đặc trưng hình học phức tạp từ 17 điểm khóa COCO. Toàn bộ các góc được tính bằng công thức arccos với tham số an toàn $\epsilon = 10^{-8}$ chống lỗi ZeroDivisionError, đảm bảo tính ổn định số học trong mọi điều kiện đầu vào.
    
    \item \textbf{Về triển khai thực tế}: Triển khai hệ thống hoàn chỉnh sẵn sàng sử dụng với giao diện PyQt5 thân thiện, module cảnh báo Telegram với cooldown 10 giây, và cơ chế sliding window stride=15, maxlen=60.
    
    \item \textbf{Về tối ưu và ổn định}: Áp dụng chiến lược Data Augmentation Isolation tránh data leakage; hàm temporal\_subsample với quy trình 3 bước (truncate $\rightarrow$ subsample $\rightarrow$ pad) đảm bảo đầu ra luôn có shape (60, 60) cố định với \texttt{assert}; kỹ thuật quản lý bộ nhớ Kaggle (\texttt{cap.release()}, \texttt{del}, \texttt{gc.collect()}) ngăn chặn memory leak trong quá trình huấn luyện dài.
\end{enumerate}

\section{Ý nghĩa khoa học và thực tiễn}
\label{sec:y_nghia_khoa_hoc_va_thuc_tien}

\textbf{Ý nghĩa khoa học}: Luận văn đóng góp vào hướng nghiên cứu phát hiện té ngã dựa trên khung xương bằng cách đề xuất phương pháp tiếp cận lai mới kết hợp Pose Estimation với Hybrid Transformer. Kiến trúc này vượt qua các hạn chế của phương pháp truyền thống (CNN/LSTM) về khả năng học phụ thuộc dài hạn, chi phí tính toán, và thiếu cơ chế attention toàn cục. Hệ thống đặc trưng PIFR với 9 góc hình học được công thức hóa rõ ràng, có thể tái sử dụng và mở rộng cho các bài toán thị giác máy tính liên quan đến phân tích tư thế con người.

\textbf{Ý nghĩa thực tiễn}: Hệ thống có tiềm năng ứng dụng trực tiếp trong thực tiễn tại Việt Nam: (1) lắp đặt tại các cơ sở chăm sóc người cao tuổi, nhà dưỡng lão, bệnh viện -- nơi cần giám sát liên tục nhiều người cao tuổi cùng lúc; (2) triển khai tại hộ gia đình có người cao tuổi sống một mình, với cảnh báo qua Telegram đến người thân; (3) kiến trúc nhẹ phù hợp triển khai trên các thiết bị Edge AI có giá thành thấp như NVIDIA Jetson Nano hoặc Raspberry Pi; (4) bảo vệ quyền riêng tư người cao tuổi vì hệ thống chỉ lưu trữ vector khung xương, không lưu trữ ảnh/video thô.

\section{Cấu trúc khóa luận}
\label{sec:cau_truc_khoa_luan}

\noindent\hspace{1.2em}
Nội dung khóa luận được tổ chức thành bốn chương chính như sau:

\begin{itemize}

    \item Chương ~\ref{chap:introduction} trình bày bài toán phát hiện té ngã, tầm quan trọng của vấn đề trong bối cảnh già hóa dân số, mục tiêu nghiên cứu, phạm vi thực hiện và các đóng góp chính của đề tài.

    \item Chương ~\ref{chap:theoretical_background} trình bày các nền tảng lý thuyết về phát hiện té ngã dựa trên khung xương, tổng quan các phương pháp hiện có, kiến trúc mô hình YOLOv11l-Pose và cơ chế hoạt động của Transformer Encoder.

    \item Chương ~\ref{chap:experiments} trình bày chi tiết kiến trúc tổng thể của hệ thống HybridFallTransformer, bao gồm module trích xuất khung xương YOLOv11l-Pose, module biến đổi đặc trưng PIFR, bộ mã hóa Hybrid Transformer; đồng thời trình bày thiết lập thực nghiệm, kết quả đánh giá hiệu năng, phân tích kết quả và so sánh với các phương pháp khác, cùng với đánh giá khả năng triển khai trên thiết bị biên Edge AI.

    \item Chương ~\ref{chap:conclusion} tổng kết các kết quả đạt được, đóng góp chính của đề tài, phân tích những hạn chế còn tồn tại và đề xuất các hướng nghiên cứu, cải tiến trong tương lai.

\end{itemize}
