import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tkinter import filedialog as fd, messagebox
from scipy.signal import convolve2d

def main():
    main = tk.Tk()
    main.geometry('1000x600')
    main.resizable(False, False)
    main.title("Canny Edge Detector - NHÓM 7")

    default_kernel_size = 7
    default_sigma = 1
    kernel_size_var = tk.StringVar(value=str(default_kernel_size))
    sigma_var = tk.StringVar(value=str(default_sigma))
    kernel_size_var_cam = tk.StringVar(value=str(default_kernel_size))
    sigma_var_cam = tk.StringVar(value=str(default_sigma))

    def show_frame(frame):
        frame.tkraise()

    def pickFileImage():
        global img_tk, gray_np
        filetypes = (
            ('Image files', '*.jpg *.png *.jpeg *.gif'),
            ('All files', '*.*')
        )
        f = fd.askopenfile(filetypes=filetypes)
        if f:
            img_path = f.name
            img = Image.open(img_path)
            img = img.resize((400, 300))
            img_tk = ImageTk.PhotoImage(img)

            gray = img.convert('L')
            gray_np = np.array(gray)
            canny_image()

            labelImage1_input.config(image=img_tk)
            labelImage1_input.image = img_tk

            text1.delete('1.0', tk.END)
            text1.insert(tk.END, img_path)

    def showCamera():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return

        def update_frame():
            ret, frame = cap.read()
            if not ret:
                return

            global gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            edges = canny_camera(gray)
            edges = np.uint8(edges)

            cam = Image.fromarray(edges)
            camtk = ImageTk.PhotoImage(cam)
            labelCamera22.config(image=camtk)
            labelCamera22.image = camtk

            labelCamera22.after(10, update_frame)

        update_frame()

        def on_closing():
            cap.release()
            cv2.destroyAllWindows()
            main.destroy()

        main.protocol("WM_DELETE_WINDOW", on_closing)

    def canny_camera(*args):
        kernel_size = int(kernel_size_var_cam.get() or default_kernel_size)
        sigma = float(sigma_var_cam.get() or default_sigma)

        gaussian_output = gaussian_smoothing(gray, kernel_size=kernel_size, sigma=sigma)


        return gaussian_output

    def canny_image(*args):
        # Bước 1:
        kernel_size = int(kernel_size_var.get() or default_kernel_size)
        sigma = float(sigma_var.get() or default_sigma)

        gaussian_output = gaussian_smoothing(gray_np, kernel_size=kernel_size, sigma=sigma)
        print('kernel_size: ' + str(kernel_size) + ' sigma: ' + str(sigma))

        #---Xử lý hiển thị---
        gaussian_image = Image.fromarray(np.uint8(gaussian_output))
        photogaussian = ImageTk.PhotoImage(gaussian_image)

        #---Cập nhật hình ảnh trên giao diện--
        labelImage1_input1.config(image=img_tk)
        labelImage1_input1.image = img_tk
        labelImage2_output1.config(image=photogaussian)
        labelImage2_output1.image = photogaussian

        #Bước 2
        gradient_magnitude, gradient_angle = gradient_operation(gaussian_output)

        #Xử lý hình ảnh hiển thị
        gradient_magnitude_image = Image.fromarray(np.uint8(gradient_magnitude))
        photogradient_magnitude = ImageTk.PhotoImage(gradient_magnitude_image)

        labelImage2_input2.config(image=photogaussian)
        labelImage2_input2.image = photogaussian

        labelImage3_output2.config(image=photogradient_magnitude)
        labelImage3_output2.image = photogradient_magnitude

    #Bước 1:
    def gaussian_smoothing(img, kernel_size, sigma):
        m, n = kernel_size, kernel_size
        gaussian_kernel = np.zeros((m, n))
        m = m // 2
        n = n // 2
        for x in range(-m, m + 1):
            for y in range(-n, n + 1):
                x1 = (sigma ** 2) * (2 * np.pi)
                x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                gaussian_kernel[x + m, y + n] = (1 / x1) * x2
        gaussian_kernel /= np.sum(gaussian_kernel)

        return convolve2d(img, gaussian_kernel, mode='same', boundary='symm')

    #Bước 2
    def gradient_operation(smoothed_image):
        sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        horizontal_gradient = convolve2d(smoothed_image, sobel_gx)
        vertical_gradient = convolve2d(smoothed_image, sobel_gy)

        gradient_magnitude = np.sqrt(horizontal_gradient ** 2 + vertical_gradient ** 2)
        gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255 # chuan hoa
        gradient_angle = np.degrees(np.arctan2(vertical_gradient, horizontal_gradient))

        return gradient_magnitude, gradient_angle

    #Theo dõi và gọi hàm
    kernel_size_var.trace_add("write", canny_image)
    sigma_var.trace_add("write", canny_image)

    kernel_size_var_cam.trace_add("write", canny_camera)
    sigma_var_cam.trace_add("write", canny_camera)

    def luu_image():
        kernel_size = int(kernel_size_var.get() or default_kernel_size)
        sigma = float(sigma_var.get() or default_sigma)

        print("kernel_size: ", kernel_size)
        print("sigma1: ", sigma)

        kernel_size_var.set(str(kernel_size))
        sigma_var.set(str(sigma))

        show_frame(frame3)

    def luu_camera():
        kernel_size = int(kernel_size_var_cam.get() or default_kernel_size)
        sigma = float(sigma_var_cam.get() or default_sigma)

        print("kernel_size: ", kernel_size)
        print("sigma1: ", sigma)

        kernel_size_var_cam.set(str(kernel_size))
        sigma_var_cam.set(str(sigma))

        show_frame(frame4)

    frame1 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame2 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame3 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame4 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame5 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame6 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame7 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame8 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame9 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame10 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')

    for frame in (frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9, frame10):
        frame.grid(row=0, column=0, sticky='nsew')

    img = Image.open("background.png")
    img = img.resize((1000, 600))
    photo = ImageTk.PhotoImage(img)

    imglg = Image.open("Hoc_vien_Hang_khongVN.PNG")
    imglg = imglg.resize((80, 64))
    photolg = ImageTk.PhotoImage(imglg)

    imgright_icon = Image.open("icon_right.png")
    imgright_icon = imgright_icon.resize((30, 30))
    right_icon = ImageTk.PhotoImage(imgright_icon)

    # Thiết kế Frame 1
    labelImg = tk.Label(frame1, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    labelImglg = tk.Label(frame1, image=photolg, bg='#BFDDE5')
    labelImglg.place(relx=0, rely=0, anchor='nw')

    label1 = tk.Label(frame1, text="Tiểu luận về Xử lý Ảnh và Thị giác Máy tính", font=('Arial', 20, 'bold'), bg='#c5d1dd', fg='#007865')
    label1.place(relx=0.5, rely=0.4, anchor='center')

    label2 = tk.Label(frame1, text="Bạn đã sẵn sàng khám phá Canny Edge Detector cùng NHÓM 7?", font=('Arial', 20, 'bold'), bg='#c5d1dd', fg='#007865')
    label2.place(relx=0.5, rely=0.5, anchor='center')

    button1 = tk.Button(frame1, text="SẴN SÀNG", command=lambda: show_frame(frame2), bg='#c5d1dd', fg='#007865', font=('Arial', 16, 'bold'), width=15, height=2, relief='flat', borderwidth=0)
    button1.place(relx=0.5, rely=0.62, anchor='center')

    # Thiết kế Frame 2
    labelImg = tk.Label(frame2, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    labelImglg = tk.Label(frame2, image=photolg, bg='#BFDDE5')
    labelImglg.place(relx=0, rely=0, anchor='nw')

    label1 = tk.Label(frame2, text="Bạn muốn chỉnh sửa nhận dạng đường biên bằng cách nào?", font=('Arial', 20, 'bold'), bg='#c5d1dd', fg='#007865')
    label1.place(relx=0.5, rely=0.45, anchor='center')

    button1 = tk.Button(frame2, text="Hình ảnh", command=lambda: show_frame(frame3), bg='#c5d1dd', fg='#007865', font=('Arial', 16, 'bold'), width=15, height=2, relief='flat', borderwidth=0)
    button1.place(relx=0.35, rely=0.57, anchor='center')

    button2 = tk.Button(frame2, text="Camera", command=lambda: show_frame(frame4), bg='#c5d1dd', fg='#007865', font=('Arial', 16, 'bold'), width=15, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.65, rely=0.57, anchor='center')

    # Thiết kế Frame 3
    labelImg = tk.Label(frame3, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame3, text="Hình ảnh", font=('Arial', 20, 'bold'), bg='#d0e7e5', fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    button1 = tk.Button(frame3, text="Quay lại", command=lambda: show_frame(frame2), bg='#fff', fg='#007865', font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    button2 = tk.Button(frame3, text="Xem chi tiết", command=lambda: show_frame(frame5), bg='#fff', fg='#007865', font=('Arial', 10, 'normal'), width=13, height=1, relief='flat', borderwidth=0)
    button2.place(relx=0.84, rely=0.05, anchor='nw')

    imgimage_icon = Image.open("icon_folder.png")
    imgimage_icon = imgimage_icon.resize((17, 17))
    image_icon = ImageTk.PhotoImage(imgimage_icon)

    text1 = tk.Text(frame3, width=30, height=1, font=('Arial', 17, 'bold'), fg="#007865", bg="white", selectbackground='#007865', wrap='none', relief='flat', padx=5, pady=5)
    text1.place(relx=0.35, rely=0.2, anchor='center')
    button3 = tk.Button(frame3, text="Chọn file",image=image_icon, compound='left', command=lambda: pickFileImage(), bg='#007865', fg='white', font=('Arial', 16, 'bold'), relief='flat', borderwidth=0, padx=10)
    button3.place(relx=0.62, rely=0.2, anchor='center')

    def kiemtra_image():
        if text1.get('1.0', tk.END).strip():
            show_frame(frame9)
        else:
            messagebox.showinfo("Thông báo", "Vui lòng chọn ảnh trước khi thay đổi!")

    button4 = tk.Button(frame3, text="Thay đổi chỉ số", command=kiemtra_image, bg='#007865', fg='#fff',
                        font=('Arial', 13, 'bold'), relief='flat', borderwidth=0, padx=7)
    button4.place(relx=0.77, rely=0.2, anchor='center')

    labelImage1 = tk.Label(frame3, text="Hình của bạn", font=('Arial', 16, 'bold'), bg='#c5d1dd', fg='#007865')
    labelImage1.place(relx=0.25, rely=0.35, anchor='center')
    frameImage1 = tk.Frame(frame3, width=400, height=300, bg='#fff')
    frameImage1.place(relx=0.25, rely=0.67, anchor='center')
    labelImage1_input = tk.Label(frameImage1)
    labelImage1_input.place(relwidth=1, relheight=1)

    labelImage2 = tk.Label(frame3, text="Hình đã chỉnh", font=('Arial', 16, 'bold'), bg='#c8d3e0', fg='#007865')
    labelImage2.place(relx=0.75, rely=0.35, anchor='center')
    frameImage2 = tk.Frame(frame3, width=400, height=300, bg='#fff')
    frameImage2.place(relx=0.75,rely=0.67, anchor='center')

    # Thiết kế Frame 4
    labelImg = tk.Label(frame4, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame4, text="Camera", font=('Arial', 20, 'bold'), bg='#d0e7e5', fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    button1 = tk.Button(frame4, text="Quay lại", command=lambda: show_frame(frame2), bg='#fff', fg='#007865', font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    imgcamera_icon = Image.open("icon_camera.png")
    imgcamera_icon = imgcamera_icon.resize((17, 17))
    camera_icon = ImageTk.PhotoImage(imgcamera_icon)
    button2 = tk.Button(frame4, text="Mở Camera", image=camera_icon, compound='left',command=lambda: showCamera(),  bg='#007865', fg='white', font=('Arial', 16, 'bold'), relief='flat', borderwidth=0, padx=10)
    button2.place(relx=0.5, rely=0.2, anchor='center')

    button4 = tk.Button(frame4, text="Thay đổi chỉ số", command=lambda: show_frame(frame10), bg='#007865', fg='#fff',
                        font=('Arial', 13, 'bold'), relief='flat', borderwidth=0, padx=7)
    button4.place(relx=0.77, rely=0.2, anchor='center')

    labelCamera2 = tk.Label(frame4, text="Camera đã chỉnh", font=('Arial', 16, 'bold'), bg='#c8d3e0', fg='#007865')
    labelCamera2.place(relx=0.5, rely=0.3, anchor='center')
    frameCamera2 = tk.Frame(frame4, width=500, height=350, bg='#fff')
    frameCamera2.place(relx=0.5,rely=0.65, anchor='center')
    labelCamera22 = tk.Label(frameCamera2)
    labelCamera22.place(relwidth=1, relheight=1)

    # Thiết kế Frame 5
    labelImg = tk.Label(frame5, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame5, text="Các Bước Tạo Ra Một Ảnh Đường Biên", font=('Arial', 20, 'bold'), bg='#d0e7e5', fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    label1 = tk.Label(frame5, text="BƯỚC 1", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label1.place(relx=0.5, rely=0.15, anchor='center')

    label1 = tk.Label(frame5, text="Làm mờ và mịn ảnh bằng bộ lọc Gaussian", font=('Arial', 17, 'bold'), bg='#D3DCE7', fg='#296958')
    label1.place(relx=0.5, rely=0.23, anchor='center')

    label2 = tk.Label(
        frame5,
        text="Bước này nhằm giảm thiểu nhiễu và làm nổi bật các cạnh quan trọng. Phân phối chuẩn Gaussian, giúp làm mượt ảnh, từ đó nâng cao độ chính xác trong phát hiện cạnh.",
        font=('Arial', 15, 'normal'),bg='#C4D4DD',fg='#296958',wraplength=620,justify='center'
    )
    label2.place(relx=0.2, rely=0.3, anchor='nw')

    button1 = tk.Button(frame5, text="Quay lại", command=lambda: show_frame(frame3), bg='#fff', fg='#007865', font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    button2 = tk.Button(frame5, text="Bước tiếp theo", command=lambda: show_frame(frame6), bg='#007865', fg='#fff', font=('Arial', 12, 'bold'), width=13, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.84, rely=0.3, anchor='nw')

    frameImage1 = tk.Frame(frame5, width=350, height=250, bg='#fff')
    frameImage1.place(relx=0.1, rely=0.53, anchor='nw')
    labelImage1_input1 = tk.Label(frameImage1)
    labelImage1_input1.place(relwidth=1, relheight=1)

    labelright = tk.Label(frame5, image=right_icon, compound='left', relief='flat', borderwidth=0, padx=10)
    labelright.place(relx=0.5, rely=0.75, anchor='center')

    frameImage2 = tk.Frame(frame5, width=350, height=250, bg='#fff')
    frameImage2.place(relx=0.55,rely=0.53, anchor='nw')
    labelImage2_output1 = tk.Label(frameImage2)
    labelImage2_output1.place(relwidth=1, relheight=1)

    # Thiết kế Frame 6
    labelImg = tk.Label(frame6, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame6, text="Các Bước Tạo Ra Một Ảnh Đường Biên", font=('Arial', 20, 'bold'), bg='#d0e7e5',
                      fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    label1 = tk.Label(frame6, text="BƯỚC 2", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label1.place(relx=0.5, rely=0.15, anchor='center')

    label1 = tk.Label(frame6, text=" Tính toán Gradient với bộ lọc Sobel", font=('Arial', 17, 'bold'), bg='#D3DCE7',
                      fg='#296958')
    label1.place(relx=0.5, rely=0.23, anchor='center')

    label2 = tk.Label(
        frame6,
        text="Bước này tạo ra ảnh gradient,thể hiện độ lớn và hướng của các cạnh trong ảnh đồng thời hỗ trợ bước 3 trong quá trình phát hiện cạnh.",
        font=('Arial', 15, 'normal'),bg='#C4D4DD',fg='#296958',wraplength=620,justify='center'
    )
    label2.place(relx=0.2, rely=0.3, anchor='nw')


    button1 = tk.Button(frame6, text="Quay lại", command=lambda: show_frame(frame3), bg='#fff', fg='#007865',
                        font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    button2 = tk.Button(frame6, text="Bước tiếp theo", command=lambda: show_frame(frame7), bg='#007865', fg='#fff',
                        font=('Arial', 12, 'bold'), width=13, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.84, rely=0.3, anchor='nw')
    button2 = tk.Button(frame6, text="Bước trước", command=lambda: show_frame(frame5), bg='#007865', fg='#fff',
                        font=('Arial', 12, 'bold'), width=13, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.04, rely=0.3, anchor='nw')

    frameImage1 = tk.Frame(frame6, width=350, height=250, bg='#fff')
    frameImage1.place(relx=0.1, rely=0.53, anchor='nw')
    labelImage2_input2 = tk.Label(frameImage1)
    labelImage2_input2.place(relwidth=1, relheight=1)

    labelright = tk.Label(frame6, image=right_icon, compound='left', relief='flat', borderwidth=0, padx=10)
    labelright.place(relx=0.5, rely=0.75, anchor='center')

    frameImage2 = tk.Frame(frame6, width=350, height=250, bg='#fff')
    frameImage2.place(relx=0.55, rely=0.53, anchor='nw')
    labelImage3_output2 = tk.Label(frameImage2)
    labelImage3_output2.place(relwidth=1, relheight=1)

    # Thiết kế Frame 7
    labelImg = tk.Label(frame7, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame7, text="Các Bước Tạo Ra Một Ảnh Đường Biên", font=('Arial', 20, 'bold'), bg='#d0e7e5',
                      fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    label1 = tk.Label(frame7, text="BƯỚC 3", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label1.place(relx=0.5, rely=0.15, anchor='center')

    button1 = tk.Button(frame7, text="Quay lại", command=lambda: show_frame(frame3), bg='#fff', fg='#007865',
                        font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    button2 = tk.Button(frame7, text="Bước tiếp theo", command=lambda: show_frame(frame8), bg='#007865', fg='#fff',
                        font=('Arial', 12, 'bold'), width=13, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.84, rely=0.3, anchor='nw')
    button2 = tk.Button(frame7, text="Bước trước", command=lambda: show_frame(frame6), bg='#007865', fg='#fff',
                        font=('Arial', 12, 'bold'), width=13, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.04, rely=0.3, anchor='nw')

    frameImage1 = tk.Frame(frame7, width=350, height=250, bg='#fff')
    frameImage1.place(relx=0.1, rely=0.53, anchor='nw')
    labelImage1_input3 = tk.Label(frameImage1)
    labelImage1_input3.place(relwidth=1, relheight=1)

    labelright = tk.Label(frame7, image=right_icon, compound='left', relief='flat', borderwidth=0, padx=10)
    labelright.place(relx=0.5, rely=0.75, anchor='center')

    frameImage2 = tk.Frame(frame7, width=350, height=250, bg='#fff')
    frameImage2.place(relx=0.55, rely=0.53, anchor='nw')
    labelImage22 = tk.Label(frameImage2)
    labelImage22.place(relwidth=1, relheight=1)

    # Thiết kế Frame 8
    labelImg = tk.Label(frame8, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame8, text="Các Bước Tạo Ra Một Ảnh Đường Biên", font=('Arial', 20, 'bold'), bg='#d0e7e5',
                      fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    label2 = tk.Label(frame8, text="BƯỚC 4", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label2.place(relx=0.5, rely=0.15, anchor='center')

    button1 = tk.Button(frame8, text="Quay lại", command=lambda: show_frame(frame3), bg='#fff', fg='#007865',
                        font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    button2 = tk.Button(frame8, text="Bước trước", command=lambda: show_frame(frame7), bg='#007865', fg='#fff',
                        font=('Arial', 12, 'bold'), width=13, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.04, rely=0.3, anchor='nw')

    frameImage1 = tk.Frame(frame8, width=350, height=250, bg='#fff')
    frameImage1.place(relx=0.1, rely=0.53, anchor='nw')
    labelImage1_input4 = tk.Label(frameImage1)
    labelImage1_input4.place(relwidth=1, relheight=1)

    labelright = tk.Label(frame8, image=right_icon, compound='left', relief='flat', borderwidth=0, padx=10)
    labelright.place(relx=0.5, rely=0.75, anchor='center')

    frameImage2 = tk.Frame(frame8, width=350, height=250, bg='#fff')
    frameImage2.place(relx=0.55, rely=0.53, anchor='nw')
    labelImage22 = tk.Label(frameImage2)
    labelImage22.place(relwidth=1, relheight=1)

    # Thiết kế Frame 9
    labelImg = tk.Label(frame9, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame9, text="THÔNG TIN CHỈ SỐ", font=('Arial', 20, 'bold'), bg='#d0e7e5',
                      fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    label2 = tk.Label(frame9, text="Bước 1", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label2.place(relx=0.5, rely=0.15, anchor='center')

    label1 = tk.Label(frame9, text="Kernel Size: ", font=('Arial', 15, 'bold'), bg='#D2D7E1', fg='#007865')
    label1.place(relx=0.3, rely=0.22, anchor='center')
    entry1 = tk.Entry(frame9,textvariable = kernel_size_var, font=('Arial',15,'normal'), fg='#007865', width=10, justify='center')
    entry1.place(relx=0.425, rely=0.22, anchor='center')

    label2 = tk.Label(frame9, text="Sigma: ", font=('Arial', 15, 'bold'), bg='#D2D7E1', fg='#007865')
    label2.place(relx=0.6, rely=0.22, anchor='center')
    entry2 = tk.Entry(frame9,textvariable = sigma_var, font=('Arial',15,'normal'), fg='#007865' ,width=10, justify='center')
    entry2.place(relx=0.7, rely=0.22, anchor='center')

    button2 = tk.Button(frame9, text="LƯU", command=luu_image, bg='#007865', fg='#fff',
                        font=('Arial', 12, 'bold'), width=13, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.5, rely=0.45, anchor='center')

    # Thiết kế Frame 10
    labelImg = tk.Label(frame10, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame10, text="THÔNG TIN CHỈ SỐ", font=('Arial', 20, 'bold'), bg='#d0e7e5',
                      fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    label2 = tk.Label(frame10, text="Bước 1", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label2.place(relx=0.5, rely=0.15, anchor='center')

    label1 = tk.Label(frame10, text="Kernel Size: ", font=('Arial', 15, 'bold'), bg='#D2D7E1', fg='#007865')
    label1.place(relx=0.3, rely=0.22, anchor='center')
    entry1 = tk.Entry(frame10,textvariable = kernel_size_var_cam, font=('Arial',15,'normal'), fg='#007865', width=10, justify='center')
    entry1.place(relx=0.425, rely=0.22, anchor='center')

    label2 = tk.Label(frame10, text="Sigma: ", font=('Arial', 15, 'bold'), bg='#D2D7E1', fg='#007865')
    label2.place(relx=0.6, rely=0.22, anchor='center')
    entry2 = tk.Entry(frame10,textvariable = sigma_var_cam, font=('Arial',15,'normal'), fg='#007865' ,width=10, justify='center')
    entry2.place(relx=0.7, rely=0.22, anchor='center')

    button2 = tk.Button(frame10, text="LƯU", command=luu_camera, bg='#007865', fg='#fff',
                        font=('Arial', 12, 'bold'), width=13, height=2, relief='flat', borderwidth=0)
    button2.place(relx=0.5, rely=0.45, anchor='center')

    show_frame(frame1)
    main.mainloop()

if __name__ == "__main__":
    main()
