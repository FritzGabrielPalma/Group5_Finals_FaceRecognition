# 𝐆𝐫𝐨𝐮𝐩𝟓_𝐅𝐢𝐧𝐚𝐥𝐬_𝐅𝐚𝐜𝐞𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧
𝑨𝒖𝒕𝒉𝒐𝒓/𝒔: 𝘑𝘰𝘩𝘯 𝘙𝘦𝘺 𝘋𝘦𝘤𝘢𝘯𝘰, 𝘚𝘩𝘦𝘳𝘪𝘭𝘺𝘯 𝘎𝘰𝘯𝘻𝘢𝘭𝘦𝘴, 𝘢𝘯𝘥 𝘍𝘳𝘪𝘵𝘻 𝘎𝘢𝘣𝘳𝘪𝘦𝘭 𝘗𝘢𝘭𝘮𝘢

H&M, short for Hennes & Mauritz, is a Swedish multinational fashion retailer renowned for its fast-fashion approach and affordable, trendy clothing. Established in 1947, the brand has grown into one of the world's largest and most recognizable fashion retailers, with a vast global presence. H&M is celebrated for its ability to swiftly translate runway trends into accessible and stylish pieces for the mass market. The company is committed to sustainability, with initiatives such as garment recycling and the use of organic materials. H&M's diverse product range spans clothing, accessories, and footwear, making fashion accessible to a broad demographic while consistently adapting to evolving style preferences.


For this task, the group is tasked to develop a face recognition program that can identify both unknown and known faces. This activity must be aligned with the past activity that this made of making an interactive Dashboard for the brand of H&M or Hennes and Mauritz. The group is to identify faces that are wearing H&M products, as well as those faces that are unknown but still wearing H&M.

The group used the following codes provided below:

### 𝐈𝐦𝐩𝐨𝐫𝐭𝐢𝐧𝐠 𝐈𝐦𝐚𝐠𝐞𝐬 𝐟𝐫𝐨𝐦 𝐆𝐢𝐭𝐡𝐮𝐛 𝐚𝐧𝐝 𝐈𝐧𝐬𝐭𝐚𝐥𝐥𝐢𝐧𝐠 𝐅𝐚𝐜𝐞_𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧
    !git clone https://github.com/FritzGabrielPalma/Group5_Finals_FaceRecognition.git
    !pip install face_recognition
    %cd Group5_Finals_FaceRecognition

### 𝐄𝐧𝐜𝐨𝐝𝐢𝐧𝐠 𝐏𝐫𝐨𝐟𝐢𝐥𝐞𝐬 𝐔𝐬𝐢𝐧𝐠 𝐊𝐧𝐨𝐰𝐧 𝐅𝐚𝐜𝐞 𝐈𝐦𝐚𝐠𝐞𝐬
    import face_recognition
    import numpy as np
    from google.colab.patches import cv2_imshow
    import cv2
    
    # Creating the encoding profiles
    face_1 = face_recognition.load_image_file("Akira Morishita.jpg")
    face_1_encoding = face_recognition.face_encodings(face_1)[0]
    
    face_2 = face_recognition.load_image_file("Gelo Rivera.jpg")
    face_2_encoding = face_recognition.face_encodings(face_2)[0]
    
    face_3 = face_recognition.load_image_file("JL Toreliza.jpg")
    face_3_encoding = face_recognition.face_encodings(face_3)[0]
    
    face_4 = face_recognition.load_image_file("Mikki Claver.jpg")
    face_4_encoding = face_recognition.face_encodings(face_4)[0]
    
    face_5 = face_recognition.load_image_file("Nate Porcalla.jpg")
    face_5_encoding = face_recognition.face_encodings(face_5)[0]
    
    known_face_encodings = [
                            face_1_encoding,
                            face_2_encoding,
                            face_3_encoding,
                            face_4_encoding,
                            face_5_encoding
    ]
    
    known_face_names = [
                        "Akira Morishita",
                        "Gelo Rivera",
                        "JL Toreliza",
                        "Mikki Claver",
                        "Nate Porcalla",
    ]

### 𝐑𝐮𝐧𝐧𝐢𝐧𝐠 𝐨𝐟 𝐅𝐚𝐜𝐞 𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧 𝐨𝐧 𝐭𝐡𝐞 𝐀𝐦𝐛𝐚𝐬𝐬𝐚𝐝𝐨𝐫𝐬 𝐨𝐟 𝐇&𝐌
> Provided below is a Python code that performs face recognition on an image using the face_recognition library and OpenCV. It loads an unknown image, detects faces, and compares their encodings with a set of known face encodings. The code then draws rectangles around recognized faces, annotates them with corresponding names, and displays the modified image, showcasing the results of the face recognition process.
> 
> For the face recognition of the ambassadors of H&M the group utilized this code both for 𝐊𝐧𝐨𝐰𝐧 and 𝐔𝐧𝐤𝐧𝐨𝐰𝐧 identities intended for this activity:

        file_name = " "
        unknown_image = face_recognition.load_image_file(file_name)
        unknown_image_to_draw = cv2.imread(file_name)
        
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        
        for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
          matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
          name = "Unknown"
        
          face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
          best_match_index = np.argmin(face_distances)
          if matches[best_match_index]:
            name = known_face_names[best_match_index]
          cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
          cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
        
        cv2_imshow(unknown_image_to_draw)
>
>         
## 𝐆𝐫𝐨𝐮𝐩 𝐏𝐡𝐨𝐭𝐨 𝐨𝐟 𝐁𝐘𝐆𝐎 𝐚𝐬 𝐀𝐦𝐛𝐚𝐬𝐬𝐚𝐝𝐨𝐫𝐬 𝐨𝐟 𝐇&𝐌
<p align="center">
  <img width="590" height="700" src="https://github.com/John-Rey-Decano/Group5_Finals_FaceRecognition/assets/143807174/155c231b-ed60-42fc-b8d7-fc6a5998da2b">
</p>


> The P-pop group members Gelo, Akira, JL, Mikki, and Nate were announced as the latest ambassadors of the global fashion giant H&M last February 17, 2022, by L'Officiel Philippines. The global fashion brand launches the MUSIC x ME campaign to promote and spotlight up-and-coming OPM artists, starting with BGYO as their newest ambassador. H&M is known for incorporating music as a marketing driving force, and with the launch of the platform MUSIC x ME, the brand's goal is to engage and allow consumers to discover new artists and sound. This campaign also dives not just into the artists themselves, but the stories of the inspiration behind them to encourage other aspiring artists to explore and do the same. With BGYO's distinct identity and catchy tunes, H&M wants to "go behind the music", and show the narrative of passion and hard work that goes into what they do, how the music they've created has been considered refreshing to many, and how that applies to their styles.
>
> 

### 𝐁𝐆𝐘𝐎 𝐗 𝐅𝐚𝐜𝐞 𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧

<p align="center">
  <img width="550" height="700" src="https://github.com/John-Rey-Decano/Group5_Finals_FaceRecognition/assets/143807174/eb7b5bcf-f0fc-4e1a-b2a4-a371c0bdbcf5">
</p>

> Face recognition is a biometric technology that identifies or verifies individuals by analyzing and matching their facial features. The process involves face detection, capturing facial images, extracting distinctive features, and comparing them with pre-stored data in a database. It is used for various applications, including security, access control, and user authentication. While face recognition offers non-intrusive and convenient solutions, privacy and data security concerns have been raised due to its potential for misuse and unauthorized surveillance.
  
### 𝐁𝐘𝐆𝐎 𝐌𝐞𝐦𝐛𝐞𝐫𝐬' 𝐅𝐚𝐯𝐨𝐫𝐢𝐭𝐞 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠
>
> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟏
>
> > (Provide Explanation)
> >
> > Using the code above for face recognition and the image with the file name "𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠𝟏.𝐣𝐩𝐠" the outcome result is:
>
<p align="center">
  <img width="600" height="600" src="https://github.com/John-Rey-Decano/Group5_Finals_FaceRecognition/assets/143807174/013e7e6c-d92c-4c00-86d3-642062666c17">
</p>


> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟐
>
> > (Provide Explanation)
> >
> > Using the code above for face recognition and the image with the file name "𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠𝟐.𝐣𝐩𝐠" the outcome result is:
>
<p align="center">
  <img width="600" height="600" src="https://github.com/John-Rey-Decano/Group5_Finals_FaceRecognition/assets/143807174/bd64d139-a9ef-4344-a09d-f417b70048f5">
</p>


> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟑
>
> > (Provide Explanation)
> >
> > Using the code above for face recognition and the image with the file name "𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠𝟑.𝐣𝐩𝐠" the outcome result is:
>
<p align="center">
  <img width="600" height="600" src="https://github.com/John-Rey-Decano/Group5_Finals_FaceRecognition/assets/143807174/4cb06487-589a-4f30-a58b-680a00253abf">
</p>




> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟒
>
> > (Provide Explanation)
> >
> > Using the code above for face recognition and the image with the file name "𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠𝟒.𝐣𝐩𝐠" the outcome result is:
>
<p align="center">
  <img width="600" height="600" src="https://github.com/John-Rey-Decano/Group5_Finals_FaceRecognition/assets/143807174/44e3e7db-fb5e-43a3-8104-d2aae5ed0ecd">
</p>



> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟓
>
> > (Provide Explanation)
> >
> > Using the code above for face recognition and the image with the file name "𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠𝟓.𝐣𝐩𝐠" the outcome result is:
>
<p align="center">
  <img width="600" height="600" src="https://github.com/John-Rey-Decano/Group5_Finals_FaceRecognition/assets/143807174/d72f51a6-e7d4-4616-abda-08a75f59d489">
</p>


 


𝐔𝐧𝐤𝐧𝐨𝐰𝐧
