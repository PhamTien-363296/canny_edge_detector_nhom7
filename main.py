import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog as fd


def main():
    main = tk.Tk()
    main.geometry('1000x600')
    main.resizable(False, False)
    main.title("Canny Edge Detector - NHÓM 7")
    cap = cv2.VideoCapture(0)

    def show_frame(frame):
        frame.tkraise()

    def pickFileImage():
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

            labelImage11.config(image=img_tk)
            labelImage11.image = img_tk

            text1.delete('1.0', tk.END)
            text1.insert(tk.END, img_path)

    def showCamera():
        imagecv2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(imagecv2)

        imgtk = ImageTk.PhotoImage(img)
        labelCamera11.imgtk = imgtk
        labelCamera11.configure(image=imgtk)
        labelCamera11.after(20, showCamera)

    frame1 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame2 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame3 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame4 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame5 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame6 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame7 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')
    frame8 = tk.Frame(main, width=1000, height=600, bg='#f0f0f0')

    for frame in (frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8):
        frame.grid(row=0, column=0, sticky='nsew')

    img = Image.open("background.png")
    img = img.resize((1000, 600))
    photo = ImageTk.PhotoImage(img)

    imgright_icon = Image.open("icon_right.png")
    imgright_icon = imgright_icon.resize((30, 30))
    right_icon = ImageTk.PhotoImage(imgright_icon)

    # Thiết kế Frame 1
    labelImg = tk.Label(frame1, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame1, text="Tiểu luận về Xử lý Ảnh và Thị giác Máy tính", font=('Arial', 20, 'bold'), bg='#c5d1dd', fg='#007865')
    label1.place(relx=0.5, rely=0.4, anchor='center')

    label2 = tk.Label(frame1, text="Bạn đã sẵn sàng khám phá Canny Edge Detector cùng NHÓM 7?", font=('Arial', 20, 'bold'), bg='#c5d1dd', fg='#007865')
    label2.place(relx=0.5, rely=0.5, anchor='center')

    button1 = tk.Button(frame1, text="SẴN SÀNG", command=lambda: show_frame(frame2), bg='#c5d1dd', fg='#007865', font=('Arial', 16, 'bold'), width=15, height=2, relief='flat', borderwidth=0)
    button1.place(relx=0.5, rely=0.62, anchor='center')

    # Thiết kế Frame 2
    labelImg = tk.Label(frame2, image=photo)
    labelImg.place(relwidth=1, relheight=1)

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
    text1.place(relx=0.43, rely=0.2, anchor='center')
    button3 = tk.Button(frame3, text="Chọn file",image=image_icon, compound='left', command=lambda: pickFileImage(), bg='#007865', fg='white', font=('Arial', 16, 'bold'), relief='flat', borderwidth=0, padx=10)
    button3.place(relx=0.7, rely=0.2, anchor='center')

    labelImage1 = tk.Label(frame3, text="Hình của bạn", font=('Arial', 16, 'bold'), bg='#c5d1dd', fg='#007865')
    labelImage1.place(relx=0.25, rely=0.35, anchor='center')
    frameImage1 = tk.Frame(frame3, width=400, height=300, bg='#fff')
    frameImage1.place(relx=0.25, rely=0.67, anchor='center')
    labelImage11 = tk.Label(frameImage1)
    labelImage11.place(relwidth=1, relheight=1)

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

    labelCamera1 = tk.Label(frame4, text="Camera của bạn", font=('Arial', 16, 'bold'), bg='#c5d1dd', fg='#007865')
    labelCamera1.place(relx=0.26, rely=0.35, anchor='center')
    frameCamera1 = tk.Frame(frame4, width=400, height=300, bg='#fff')
    frameCamera1.place(relx=0.25, rely=0.67, anchor='center')
    labelCamera11 = tk.Label(frameCamera1)
    labelCamera11.place(relwidth=1, relheight=1)

    labelCamera2 = tk.Label(frame4, text="Camera đã chỉnh", font=('Arial', 16, 'bold'), bg='#c8d3e0', fg='#007865')
    labelCamera2.place(relx=0.73, rely=0.35, anchor='center')
    frameCamera2 = tk.Frame(frame4, width=400, height=300, bg='#fff')
    frameCamera2.place(relx=0.75,rely=0.67, anchor='center')

    # Thiết kế Frame 5
    labelImg = tk.Label(frame5, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame5, text="Các Bước Tạo Ra Một Ảnh Đường Biên", font=('Arial', 20, 'bold'), bg='#d0e7e5', fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    label1 = tk.Label(frame5, text="BƯỚC 1", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label1.place(relx=0.5, rely=0.15, anchor='center')

    button1 = tk.Button(frame5, text="Quay lại", command=lambda: show_frame(frame3), bg='#fff', fg='#007865', font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    button2 = tk.Button(frame5, text="Bước tiếp theo", command=lambda: show_frame(frame6), bg='#fff', fg='#007865', font=('Arial', 12, 'normal'), width=13, height=1, relief='flat', borderwidth=0)
    button2.place(relx=0.84, rely=0.3, anchor='nw')

    frameImage1 = tk.Frame(frame5, width=350, height=250, bg='#fff')
    frameImage1.place(relx=0.1, rely=0.53, anchor='nw')
    labelImage11 = tk.Label(frameImage1)
    labelImage11.place(relwidth=1, relheight=1)

    labelright = tk.Label(frame5, image=right_icon, compound='left', relief='flat', borderwidth=0, padx=10)
    labelright.place(relx=0.5, rely=0.75, anchor='center')

    frameImage2 = tk.Frame(frame5, width=350, height=250, bg='#fff')
    frameImage2.place(relx=0.55,rely=0.53, anchor='nw')
    labelImage22 = tk.Label(frameImage2)
    labelImage22.place(relwidth=1, relheight=1)

    # Thiết kế Frame 6
    labelImg = tk.Label(frame6, image=photo)
    labelImg.place(relwidth=1, relheight=1)

    label1 = tk.Label(frame6, text="Các Bước Tạo Ra Một Ảnh Đường Biên", font=('Arial', 20, 'bold'), bg='#d0e7e5',
                      fg='#007865')
    label1.place(relx=0.5, rely=0.08, anchor='center')

    label1 = tk.Label(frame6, text="BƯỚC 2", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label1.place(relx=0.5, rely=0.15, anchor='center')

    button1 = tk.Button(frame6, text="Quay lại", command=lambda: show_frame(frame3), bg='#fff', fg='#007865',
                        font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    button2 = tk.Button(frame6, text="Bước tiếp theo", command=lambda: show_frame(frame7), bg='#fff', fg='#007865',
                        font=('Arial', 12, 'normal'), width=13, height=1, relief='flat', borderwidth=0)
    button2.place(relx=0.84, rely=0.3, anchor='nw')
    button2 = tk.Button(frame6, text="Bước trước", command=lambda: show_frame(frame5), bg='#fff', fg='#007865',
                        font=('Arial', 12, 'normal'), width=13, height=1, relief='flat', borderwidth=0)
    button2.place(relx=0.04, rely=0.3, anchor='nw')

    frameImage1 = tk.Frame(frame6, width=350, height=250, bg='#fff')
    frameImage1.place(relx=0.1, rely=0.53, anchor='nw')
    labelImage11 = tk.Label(frameImage1)
    labelImage11.place(relwidth=1, relheight=1)

    labelright = tk.Label(frame6, image=right_icon, compound='left', relief='flat', borderwidth=0, padx=10)
    labelright.place(relx=0.5, rely=0.75, anchor='center')

    frameImage2 = tk.Frame(frame6, width=350, height=250, bg='#fff')
    frameImage2.place(relx=0.55, rely=0.53, anchor='nw')
    labelImage22 = tk.Label(frameImage2)
    labelImage22.place(relwidth=1, relheight=1)

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

    button2 = tk.Button(frame7, text="Bước tiếp theo", command=lambda: show_frame(frame8), bg='#fff', fg='#007865',
                        font=('Arial', 12, 'normal'), width=13, height=1, relief='flat', borderwidth=0)
    button2.place(relx=0.84, rely=0.3, anchor='nw')
    button2 = tk.Button(frame7, text="Bước trước", command=lambda: show_frame(frame6), bg='#fff', fg='#007865',
                        font=('Arial', 12, 'normal'), width=13, height=1, relief='flat', borderwidth=0)
    button2.place(relx=0.04, rely=0.3, anchor='nw')

    frameImage1 = tk.Frame(frame7, width=350, height=250, bg='#fff')
    frameImage1.place(relx=0.1, rely=0.53, anchor='nw')
    labelImage11 = tk.Label(frameImage1)
    labelImage11.place(relwidth=1, relheight=1)

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

    label1 = tk.Label(frame8, text="BƯỚC 4", font=('Arial', 20, 'bold'), bg='#CFE1E4', fg='#007865')
    label1.place(relx=0.5, rely=0.15, anchor='center')

    button1 = tk.Button(frame8, text="Quay lại", command=lambda: show_frame(frame3), bg='#fff', fg='#007865',
                        font=('Arial', 10, 'normal'), width=10, height=1, relief='flat', borderwidth=0)
    button1.place(relx=0.05, rely=0.05, anchor='nw')

    button2 = tk.Button(frame8, text="Bước trước", command=lambda: show_frame(frame7), bg='#fff', fg='#007865',
                        font=('Arial', 12, 'normal'), width=13, height=1, relief='flat', borderwidth=0)
    button2.place(relx=0.04, rely=0.3, anchor='nw')

    frameImage1 = tk.Frame(frame8, width=350, height=250, bg='#fff')
    frameImage1.place(relx=0.1, rely=0.53, anchor='nw')
    labelImage11 = tk.Label(frameImage1)
    labelImage11.place(relwidth=1, relheight=1)

    labelright = tk.Label(frame8, image=right_icon, compound='left', relief='flat', borderwidth=0, padx=10)
    labelright.place(relx=0.5, rely=0.75, anchor='center')

    frameImage2 = tk.Frame(frame8, width=350, height=250, bg='#fff')
    frameImage2.place(relx=0.55, rely=0.53, anchor='nw')
    labelImage22 = tk.Label(frameImage2)
    labelImage22.place(relwidth=1, relheight=1)
    show_frame(frame1)
    main.mainloop()

if __name__ == "__main__":
    main()