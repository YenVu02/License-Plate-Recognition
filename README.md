# License-Plate-Recognition
License Plate Recognition For Car With Python And OpenCV
####Nhận dạng biển số xe Trung Quốc bằng python3+opencv3, gồm thuật toán và giao diện client, chỉ có 2 file, surface.py là mã giao diện, Predict.py là mã thuật toán, giao diện không phải là trọng tâm nên là rất đơn giản để viết với tkinter.

### Hướng dẫn:
Phiên bản: python3.4.4, opencv3.4 và numpy1.14 và PIL5<br>
Tải xuống mã nguồn, cài đặt phiên bản python của python, numpy, opencv và PIL và chạy surface.py

### Triển khai thuật toán:
Ý tưởng của thuật toán đến từ các tài nguyên trực tuyến, đầu tiên nó sử dụng các cạnh hình ảnh và màu biển số xe để xác định vị trí biển số xe, sau đó nhận dạng các ký tự. Biển số xe được định vị theo phương pháp dự đoán, để rõ ràng, sau khi hoàn thiện mã và thử nghiệm, rất nhiều nhận xét đã được thêm vào, vui lòng tham khảo mã nguồn. Nhận dạng ký tự biển số xe cũng nằm trong phương pháp dự đoán. Vui lòng tham khảo các nhận xét trong mã nguồn. Cần lưu ý rằng thuật toán được sử dụng để nhận dạng ký tự biển số xe là SVM của opencv. Mã sử ​​dụng SVM của opencv lấy từ mẫu đi kèm với opencv. Cả lớp StatModel và lớp SVM đều là mã trong mẫu. Các mẫu đào tạo được sử dụng trong đào tạo SVM đến từ phiên bản c++ của EasyPR trên github. Do số lượng mẫu đào tạo còn hạn chế, bạn sẽ thấy có thể xảy ra lỗi trong quá trình nhận dạng ký tự biển số xe trong quá trình thử nghiệm, đặc biệt khi ký tự tiếng Trung đầu tiên xuất hiện với khả năng xảy ra lỗi cao. Trong mã nguồn, tôi đã tải các mẫu đào tạo trong EasyPR vào thư mục train\, nếu bạn muốn đào tạo lại, vui lòng giải nén trong thư mục hiện tại và xóa các tệp dữ liệu đào tạo gốc svm.dat và svmchinese.dat.

##### Hướng dẫn bổ sung: Mã thuật toán chỉ có 500 dòng, trong quá trình thử nghiệm nhận thấy các thông số của thuật toán định vị biển số xe bị ảnh hưởng bởi độ phân giải hình ảnh, độ lệch màu và khoảng cách xe (pixel của giấy phép) các biển số trong thư mục kiểm tra tương đối nhỏ và các hình ảnh khác có thể xảy ra. Do không thể nhận dạng được pixel và các vấn đề khác, nên việc xác định biển số xe bằng các pixel khác yêu cầu sửa đổi các tham số trong tệp cấu hình. Dự án này chỉ nhằm truyền cảm hứng cho những người khác và đưa ra ý tưởng ).
##### Hiệu ứng giao diện:
![]((https://github.com/YenVu02/License-Plate-Recognition/blob/main/Screenshots/3.png))
