# Attendance-system-face-recognition
The Attendance system is based on the machine learning algorithm which is to be implemented on python language and using computer/laptop camera 
for the input image of the students or a normal outer camera can also be used which has to be connected to the system which is programmed 
to handle the face recognition by implementing the Convolutional Neural Network algorithm.

Steps to Run:
1. Requires python, vscode or any compiler, mysql 
2. Install required libriaries:
   1. tkinter
   2. opencv-python
   3. pandas
   4. scikit-learn
   5. keras
   6. tensorflow
   7. numpy
   8. os
   9. csv
   10. mysql.connector
3. Download the two files in a same folder. Edit the code 1.py replacing all the paths withs appropriate path.
4. Run the 1.py
5. In the GUI that opens up, register the faces with details and it will automatically opens webcam and take pictures and stores in folder dataset. [Edit code with the folderpath]
6. Train the model with using the botton in GUI.[This trains the CNN model with the collected photos. Do this once after registering all the persons.]
7. Take attendance, and mark attendance buttons are also provided.
