# insect classication app by Pranav Shinde.
import streamlit as st
from PIL import Image
import matplotlib as plt
 
       

if __name__ == "__main__":
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Create")
    if file_uploaded is not None:    
        my_images = Image.open(file_uploaded)
        st.image(my_images, caption='Uploaded Image', use_column_width=True)
        plt.imshow(my_images)
   
    for i, image_name in enumerate(my_images):  
        if (image_name.split('.')[1] == 'jpg'):        
            image = io.imread(image_directory + image_name)        
            image = Image.fromarray(image, 'RGB')        
            image = image.resize((SIZE,SIZE)) 
            dataset.append(np.array(image))
        x = np.array(dataset)

    str=st.text_input(" Enter the directory name in which file to be stored \n ")
    i = 0
    for batch in datagen.flow(x, batch_size=16,save_to_dir= r'str',save_prefix='dr',save_format='jpg'):
        i += 1
        if i > 10:
            break
