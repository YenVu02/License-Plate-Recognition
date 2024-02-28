import tkinter as tk
from tkinter import LEFT, TOP, RIGHT
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter import ttk
import predict
import cv2
from PIL import Image, ImageTk
import threading
import time
import mysql.connector
import paho.mqtt.client as mqtt

class Surface(ttk.Frame):
    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        # Subscribe to topic on connection
        client.subscribe("topic/plate_recognition")  # Thay đổi topic thành topic bạn muốn

    def on_message(self, client, userdata, msg):
        message = msg.payload.decode()
        # Xử lý thông điệp nhận được tại đây

    def connect_mqtt_broker(self):
        client = mqtt.Client("P1")  # Tạo một client MQTT
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect("172.19.200.163", 1883, 60)  # Kết nối tới MQTT broker
        return client

    def publish_message(self, topic, message):
        client = self.connect_mqtt_broker()
        client.publish(topic, message)
        client.disconnect()  # Ngắt kết nối sau khi gửi

    def subscribe_to_topic(self, topic):
        client = self.connect_mqtt_broker()
        client.loop_forever()  # Lắng nghe liên
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {"green": ("Xanh lá", "#55FF55"), "yello": ("Vàng", "#FFFF00"), "blue": ("Xanh dương", "#6666FF"), "white": ("Trắng", "#FFFFFF")}

    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("Nhận diện biển số xe")
        win.state("zoomed")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=tk.LEFT, expand=1, fill=tk.BOTH)

        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=tk.RIGHT, expand=0)
        ttk.Label(frame_left, text='Ảnh gốc:').pack(anchor="nw")
        ttk.Label(frame_right1, text='Vị trí biển số xe:').grid(column=0, row=0, sticky=tk.W)

        from_pic_ctl = ttk.Button(frame_right2, text="Từ ảnh", width=20, command=self.from_pic)
        from_vedio_ctl = ttk.Button(frame_right2, text="Từ camera", width=20, command=self.from_vedio)
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='Kết quả nhận diện:').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        from_vedio_ctl.pack(anchor="se", pady="5")
        from_pic_ctl.pack(anchor="se", pady="5")
        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()
        self.status_label = ttk.Label(frame_right1, text="Trạng thái: Chưa xác định")  # Add a label for status
        self.status_label.grid(column=0, row=5, sticky=tk.W)  # Position the label

        self.capture_flag = False
        self.esp32_signal = False
        self.database_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '123456',
            'database': 'btl_iot'
        }
        self.create_database()
        #self.is_license_plate_in_database("test_plate_number")  # Thêm dòng in này
        #self.add_license_plate_to_database("test_plate_number")  # Thêm dòng in này
        self.capture_image()  # Thêm dòng in này
    
    def create_database(self):
        conn = mysql.connector.connect(**self.database_config)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS license_plates (
                plate_number VARCHAR(255) PRIMARY KEY
            )
        ''')
        conn.commit()
        conn.close()

    def is_license_plate_in_database(self, plate_number):
        conn = mysql.connector.connect(**self.database_config)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM license_plates WHERE plate_number=%s", (plate_number,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    
    
    def set_esp32_signal(self, signal):
        self.esp32_signal = signal
    # Thêm hàm để giảm kích thước ảnh
    def resize_image(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)

        # Thay đổi kích thước ảnh tại đây
        resized_im = im.resize((self.viewwide, self.viewhigh), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=resized_im)
        return imgtk
    # Thay đổi hàm capture_image()
      # Thay đổi hàm capture_image()
    def capture_image(self):
        if self.thread_run and self.esp32_signal:
            _, img_bgr = self.camera.read()
            
            # Thay đổi dòng này để sử dụng hàm resize_image
            self.imgtk = self.resize_image(img_bgr)
            
            self.image_ctl.configure(image=self.imgtk)
            predict_time = time.time()
            print("Trước khi gọi hàm predict")
            r, roi, color = self.predictor.predict(img_bgr)
            print("giá trị của biển số:", r)  # Thêm dòng in này
            
            self.show_roi(r, roi, color)
            predict_time = time.time()
            self.esp32_signal = False
            if r:
                plate_number = ''.join(r)
                is_in_database = self.is_license_plate_in_database(plate_number)
                check = 1
                if is_in_database:
                    check = 0
                    print(f"Biển số {plate_number} đã có trong database: {check}")
                else:
                    check = 0
                    print(f"Biển số {plate_number} chưa có trong database: {check}")
                   
    # Thay đổi hàm vedio_thread()
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            
            # Thay đổi dòng này để sử dụng hàm resize_image
            self.imgtk = self.resize_image(img_bgr)
            
            self.image_ctl.configure(image=self.imgtk)
            if time.time() - predict_time > 2:
                r, roi, color = self.predictor.predict(img_bgr)
                self.show_roi(r, roi, color)
                predict_time = time.time()
        print("run end")
    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)

        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)

            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1

            im = im.resize((wide, high), Image.ANTIALIAS)


            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    # Thay đổi hàm show_roi() trong lớp Surface như sau:
    def show_roi(self, r, roi, color):
        if r:
            plate_number = ''.join(r)
            print(f"Biển số xe: {plate_number}", end=' - ')
            #print(f"Vị trí biển số xe: {roi}")  # Print the location of the license plate
            print(f"Màu sắc: {color}", end=' - ')  # Print the color
            
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')

            self.r_ctl.configure(text=plate_number)
            self.update_time = time.time()
            
            try:
                c = self.color_transform[color]
                self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            except:
                self.color_ctl.configure(state='disabled')

            # Check if the license plate is in the database
             
            # Check if the license plate is in the database
            is_in_database = self.is_license_plate_in_database(plate_number)
            check = 1
            if is_in_database:
                check = 1
                status = "Đã đăng ký"
                print(f"Biển số xe: đã đăng kí")
            else:
                status = "Chưa đăng ký"
                check = 0
                # Cập nhật trạng thái vào self.status_labels
                self.status_label.configure(text=f"Trạng thái: {status}")
                print(f"Biển số xe: chưa đăng kí")
                # Prompt user for registration
                #register_plate = messagebox.askyesno("Đăng ký", f"Biển số {plate_number} chưa có trong cơ sở dữ liệu. Bạn có muốn đăng ký không?")
                #if register_plate:
                   # self.add_license_plate_to_database(plate_number)
                    #status = "Đã đăng ký"
                    #check = 1

       
            
            # Update the status label with the registration status
            self.status_label.configure(text=f"Trạng thái: {status}")
            
            self.publish_message("topic/plate_recognition", check)

        elif self.update_time + 8 < time.time():
            # Biển số không được xác định - cập nhật trạng thái
            status = "Chưa xác định"
            self.status_label.configure(text=f"Trạng thái: {status}")
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            self.color_ctl.configure(state='disabled')

    def from_vedio(self):
        if self.thread_run:
            return
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                mBox.showwarning('Warning', 'Failed to open camera!')
                self.camera = None
                return
        self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread_run = True

    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="Chọn ảnh nhận diện", filetypes=[("jpg images", "*.jpg"), ("PNG images", "*.png")])
        if self.pic_path:
            img_bgr = predict.imreadex(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
            for resize_rate in resize_rates:
                #print("resize_rate:", resize_rate)
                r, roi, color = self.predictor.predict(img_bgr, resize_rate)
                if r:
                    break
            self.show_roi(r, roi, color)

    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            if time.time() - predict_time > 2:
                r, roi, color = self.predictor.predict(img_bgr)
                self.show_roi(r, roi, color)
                predict_time = time.time()
        print("run end")

def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()

if __name__ == '__main__':
    win = tk.Tk()
    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()
