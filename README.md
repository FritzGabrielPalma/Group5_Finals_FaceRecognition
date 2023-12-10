# 𝐆𝐫𝐨𝐮𝐩𝟓_𝐅𝐢𝐧𝐚𝐥𝐬_𝐅𝐚𝐜𝐞𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧
𝑨𝒖𝒕𝒉𝒐𝒓/𝒔: 𝘑𝘰𝘩𝘯 𝘙𝘦𝘺 𝘋𝘦𝘤𝘢𝘯𝘰, 𝘚𝘩𝘦𝘳𝘪𝘭𝘺𝘯 𝘎𝘰𝘯𝘻𝘢𝘭𝘦𝘴, 𝘢𝘯𝘥 𝘍𝘳𝘪𝘵𝘻 𝘎𝘢𝘣𝘳𝘪𝘦𝘭 𝘗𝘢𝘭𝘮𝘢

For this task the group are task to develop a face recognition program that can identify both unkown and known faces. This activity must be aligned to the past activity that this made of making an interactive Dashboard for the brand of H&M or Hennes and Mauritz. The group is to identify faces that are wearing H&M products, as well as those faces that are unknown but still wearing H&M.

The group use the following codes provided below:

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
> 𝐆𝐫𝐨𝐮𝐩 𝐏𝐡𝐨𝐭𝐨 𝐨𝐟 𝐁𝐘𝐆𝐎 𝐚𝐬 𝐀𝐦𝐛𝐚𝐬𝐬𝐚𝐝𝐨𝐫𝐬 𝐨𝐟 𝐇&𝐌
>
> > (Provide explanation)

    file_name = "G4.jpg"
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
    
> 𝐁𝐘𝐆𝐎 𝐌𝐞𝐦𝐛𝐞𝐫𝐬' 𝐅𝐚𝐯𝐨𝐫𝐢𝐭𝐞 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠
>
> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟏
>
> > (Provide Explanation)
>
    file_name = "Clothing1.jpg"
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

> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟐
>
> > (Provide Explanation)
>
    file_name = "Clothing2.jpg"
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

> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟑
>
> > (Provide Explanation)
>
    file_name = "Clothing3.jpg"
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

> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟒
>
> > (Provide Explanation)
>
    file_name = "Clothing4.jpg"
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

> 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟓
>
> > (Provide Explanation)
> 
    file_name = "Clothing5.jpg"
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


𝐔𝐧𝐤𝐧𝐨𝐰𝐧
