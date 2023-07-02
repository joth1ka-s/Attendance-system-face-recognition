import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import pandas as pd
import cv2
import os
import csv
import numpy as np
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from datetime import date
import mysql.connector

haar_file = 'haarcascade_frontalface_default.xml'
current_date = date.today()
date_string = current_date.strftime("%Y-%m-%d")

#MySQL Connection
conn= mysql.connector.connect(host='localhost', password='jo@MYSQL0', user='root')
cur=conn.cursor(buffered=True)
#creating database(using if already exist)
try:
    cur.execute("use smartattendance")
except:
    cur.execute("create database smartattendance")
    cur.execute("use smartattendance")
#creating table(using if already exist)
try:
    cur.execute("describe studentDetails")
except:
    cur.execute("create table studentDetails(roll_no int primary key auto_increment, name varchar(20) not null,batch varchar(20) not null, email varchar(20) not null, phno varchar(20) not null)")

try:
    cur.execute("describe Attendance")
except:
    cur.execute("create table Attendance(roll_no int primary key auto_increment, name varchar(20) not null)")





def register_students():
    name=name_entry.get()
    email=email_entry.get()
    batch=batch_entry.get()
    phno=phno_entry.get()

    datasets = 'dataset'    
    sub_data = name  # Get the name entered in the Entry widget

    #store other details in database studentdetails
    cur.execute(f"insert into studentDetails(name, batch, email, phno) values ('{name}','{batch}','{email}','{phno}')")
    cur.execute(f"insert into Attendance(name) values ('{name}')")
    conn.commit()

    # Specify the path where you want to create the folder
    path = "D:\\proj\\dataset"
    folder_path = os.path.join(path, sub_data)
    os.makedirs(folder_path, exist_ok=True)

    print("Folder created at:", folder_path)

    path = os.path.join(path, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    (width, height) = (130, 100)   
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0) 
    count = 1
    while count < 101:
        print(count)
        (_, im) = webcam.read()
        cv2.imwrite('%s/%s.png' % (path,count), im)
        count += 1
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    webcam.release()
    cv2.destroyAllWindows()

def train_model():

    # Set the directory path where the images are stored
    dir_path = "D:\\proj\\dataset"
    # Create a list to store the image paths and labels
    dataset = []
    # Loop through each folder and read the images
    for folder in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):  # Check for image files
                    label = folder  # Set the label as the folder name
                    dataset.append([file_path, label])  # Add the image path and label to the list
    # Specify the file path
    csv_file_path = 'D:\\proj\\imageDataset.csv'
    # Add the image_path and label as the first rows in the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the headers
        writer.writerow(['image_path', 'label'])

        # Write the dataset rows
        writer.writerows(dataset)


    # Read dataset from csv file
    df = pd.read_csv('D:\\proj\\imageDataset.csv')
    # Initialize Haar cascade classifier
    face_cascade = cv2.CascadeClassifier("D:\\proj\\haarcascade_frontalface_default.xml")
    # Preprocess dataset
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # if no faces are detected, return None
        if len(faces) == 0:
            return None
        # get first detected face
        (x, y, w, h) = faces[0]
        # extract face from image
        face_image = image[y:y+h, x:x+w]
        # return preprocessed image
        return face_image
    # Apply preprocessing function to images in dataset
    df['image'] = df['image_path'].apply(preprocess_image)
    # Remove rows where no faces were detected
    df.dropna(inplace=True)

    # Split dataset into training, testing, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(df['image'], df['label'], test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.075, random_state=42)
    # Resize images to a fixed size
    X_train = [cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA) for image in X_train]
    X_test = [cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA) for image in X_test]
    X_val = [cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA) for image in X_val]
    # Convert training, testing, and validation sets into numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)
    # Normalize pixel values of images
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_val = X_val / 255.0
    # Print shapes of training, testing, and validation sets
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('X_val shape:', X_val.shape)
    print('y_val shape:', y_val.shape)
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Fit and transform labels for training set
    y_train_encoded = label_encoder.fit_transform(y_train)
    # Transform labels for testing and validation sets
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)
    
    y_train_one_hot = to_categorical(y_train_encoded)
    y_test_one_hot = to_categorical(y_test_encoded)
    y_val_one_hot = to_categorical(y_val_encoded)

    # Load pre-trained VGG16 model without top classifier
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    # Freeze most of the base model's layers
    for layer in base_model.layers[:-4]:
        layer.trainable = True
    folder_path = "D:\\proj\\dataset"
    # Get a list of all items (files and folders) in the specified directory
    items = os.listdir(folder_path)
    # Filter out only the folders from the list of items
    class_labels = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    count=len(class_labels)
    print(class_labels)
    # Create a new model on top of the base model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(count, activation='softmax'))
    # Compile the model with a lower learning rate
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model with the fine-tuned layers
    model.fit(X_train, y_train_one_hot, epochs=5, batch_size=16, validation_data=(X_val, y_val_one_hot))
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
    print('Test accuracy:', test_acc)
    model.save('project.h5')
    # Get a list of all items (files and folders) in the specified directory
   

    # Filter out only the folders from the list of items

    # Create a DataFrame to store the class labels
    #data = {'Name': class_labels}
    #df = pd.DataFrame(data)

    # Write the labels to an Excel file
    #file_path = 'labels.xlsx'
    #df.to_excel(file_path, index=False)
    #print("Labels have been written to", file_path)

def mark_attendance():
    #global attendance_df

    # Get the current system date
    folder_path="D:\\proj\\dataset"
    current_date = date.today()
    items = os.listdir(folder_path)
    class_labels = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    # Transform the date object to a string with the format "YYYY-MM-DD"
    date_string = current_date.strftime("%Y-%m-%d")
    # Load the trained model
    model = load_model('D:\\proj\\project.h5')

    # Read the attendance labels from the database
    cur.execute(f"SELECT name FROM attendance")
    # Fetch all the rows returned by the query
    rows = cur.fetchall()
    labels = []
    # Process the rows and store the data in the list
    for row in rows:
        name = row[0]
        labels.append(name)
    # Create a DataFrame from the retrieved data
    attendance_df = pd.DataFrame(labels, columns=[name])
    
    # Define the confidence threshold for label recognition
    confidence_threshold = 0.7

    # Define the image preprocessing function
    def preprocess_image(image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no faces are detected, return None
        if len(faces) == 0:
            return None

        # Get the first detected face
        (x, y, w, h) = faces[0]

        # Extract the face from the image
        face_image = image[y:y+h, x:x+w]

        # Return the preprocessed image
        return face_image

    # Load the Haar cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier('D:\\proj\\haarcascade_frontalface_default.xml')

    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Create a canvas to display the video stream
    video_canvas = tk.Label(markAtt_frame)
    video_canvas.grid(row=2, column=1, padx=20)

    # Create a dictionary to track label stability
    label_stability = {}

    # Create a variable to store the last stable label
    last_stable_label = None
    

    # Create a variable to store the start time of the current label stability duration
    stability_start_time = None
    
    
    # Create a variable to store the duration required for stability (in seconds)
    stability_duration = 3

    def update_frame(last_stable_label):
        global stability_start_time
        print(last_stable_label)

        # Capture the frame
        ret, frame = cap.read()

        # Preprocess the frame
        face_image = preprocess_image(frame)

        if face_image is not None:
            # Resize and normalize the image
            face_image = cv2.resize(face_image, (100, 100), interpolation=cv2.INTER_AREA)
            face_image = face_image / 255.0
            face_image = np.expand_dims(face_image, axis=0)

            # Make predictions on the preprocessed image
            predictions = model.predict(face_image)
            max_confidence = np.max(predictions)
            

            if max_confidence < confidence_threshold:
                label = "Unrecognized"
            else:
                label_index = np.argmax(predictions)
                label = class_labels[label_index]

            # Check if the predicted label is the same as the last stable label
            if label == last_stable_label:
                # Check if the stability timer has started
                if stability_start_time is None:
                    # Start the stability timer
                    stability_start_time = cv2.getTickCount()
                else:
                    # Calculate the elapsed time since the stability timer started
                    elapsed_time = (cv2.getTickCount() - stability_start_time) / cv2.getTickFrequency()

                    # Check if the stability duration has been reached
                    if elapsed_time >= stability_duration:
                        #Mark attendance for the stable label in excel
                        
                        #Mark attendance for the stable label in sql database
                        cur.execute(f"SHOW COLUMNS FROM Attendance LIKE '{date_string}'")
                        column_exists = cur.fetchone()

                        # Add the column only if it doesn't already exist
                        if not column_exists:
                            cur.execute(f"ALTER TABLE Attendance ADD COLUMN `{date_string}` VARCHAR(10) DEFAULT 'Absent'")
                        cur.execute(f"UPDATE Attendance SET `{date_string}` = 'Present' WHERE name = '{label}'")
                        conn.commit()

                        # Reset the stability timer and update the last stable label
                        stability_start_time = None
                        last_stable_label = label
            else:
                # Reset the stability timer and update the last stable label
                stability_start_time = None
                last_stable_label = label

            # Draw the bounding box around the face
            (x, y, w, h) = face_cascade.detectMultiScale(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put the label next to the bounding box
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to PIL format
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image to fit the canvas
        img = img.resize((400, 300))

        # Convert the image to Tkinter format
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the video canvas with the new frame
        video_canvas.img_tk = img_tk
        video_canvas.config(image=img_tk)

        # Call the update_frame function after 1ms (i.e., 30 frames per second)
        video_canvas.after(1, update_frame,last_stable_label)

    # Start updating the frame
    update_frame(last_stable_label)
    #attendance_df.to_excel('labels.xlsx', index=False)

def show_attendance():
    
    #mysql...
    # Build and execute the query to select rollno, name, and today's date column
    cur.execute(f"SELECT roll_no, name, `{date_string}` FROM Attendance")
    # Fetch all the rows returned by the query
    rows = cur.fetchall()
    # Create a list to store the retrieved data
    attendance_data = []
    # Process the rows and store the data in the list
    for row in rows:
        rollno = row[0]
        name = row[1]
        attendance_status = row[2]
        attendance_data.append([rollno, name, attendance_status])
    # Create a DataFrame from the retrieved data
    attendance_df = pd.DataFrame(attendance_data, columns=["rollno", "name", date_string])
    # Create a text widget to display the attendance
    text_widget = tk.Text(markAtt_frame,height=20, width=40)
    text_widget.grid(row=2, column=2,padx=20)
    # Insert the attendance data into the text widget
    text_widget.insert(tk.END, attendance_df.to_string(index=False))

    # Close the cursor and database connection
    #cur.close()
    #conn.close()

# Create the Tkinter app
app = tk.Tk()
app.title("Face Recognition App")
# Set the window dimensions
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
app.geometry(f"{screen_width}x{screen_height}")

#divide window into 2 sections
#registration frame
reg_frame = tk.LabelFrame(app, text="Student Registration",pady=60,padx=20, font= ('Helvetica 13 bold'))
reg_frame.pack(side=LEFT, expand=True, fill=BOTH)

stu_label = tk.Label(reg_frame, text="Enter the Details:")
name_label = tk.Label(reg_frame, text="Name:")
name_entry = tk.Entry(reg_frame)
email_label = tk.Label(reg_frame, text="Mail id:")
email_entry = tk.Entry(reg_frame)
phno_label=tk.Label(reg_frame, text="Phone no (parent) :")
phno_entry = tk.Entry(reg_frame)
batch_label = tk.Label(reg_frame, text="Batch :")
batch_entry = tk.Entry(reg_frame)

reg1_button = tk.Button(reg_frame, text="Register", command=register_students)
reg1_button.grid(row=7,column=1,padx= 60, pady= 8)
train_button = tk.Button(reg_frame, text="Train the model", command=train_model)
train_button.grid(row=10, column=1,pady= 80)

stu_label.grid(row=1,column=1,pady= 8)
name_label.grid(row=2, column=1, pady= 8)
name_entry.grid(row=2, column=2, pady=8)
email_label.grid(row=3, column=1, pady=8)
email_entry.grid(row=3, column=2, pady=8)
phno_label.grid(row=4, column=1, pady=8)
phno_entry.grid(row=4, column=2, pady=8)
batch_label.grid(row=5, column=1, pady=8)
batch_entry.grid(row=5, column=2, pady=8)

#attendance marking frame
markAtt_frame = tk.LabelFrame(app, text="Attendance",padx=100, pady=50,font= ('Helvetica 13 bold'))
markAtt_frame.pack(side=LEFT, expand=True, fill=BOTH)
mark_button = tk.Button(markAtt_frame, text="Mark Attendance", command=mark_attendance)
mark_button.grid(row=1, column=1)
show_button = tk.Button(markAtt_frame, text="Show Attendance", command=show_attendance)
show_button.grid(row=1, column=2)

# Start the Tkinter event loop
app.mainloop()