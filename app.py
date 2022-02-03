import streamlit as st
import time
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
import cv2

def ain(save_image_path):
    
    img=load_img(save_image_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img_file=np.expand_dims(img,[0])
    
    st.write("SIFT algorithm is using")
    '''
    sift = cv2.SIFT_create()
    kry_points,descriptors = sift.detectAndCompute(img_file,None)
    final_image = cv2.drawKeypoints(img_file,kry_points,None)
    st.write("final image after SIFT")
    st.image(final_image,use_column_width=False)'''
    
        

def waitt():
    with st.spinner(text='In progress'):
        
        time.sleep(100)
        
def aod(img_file):
    st.write('Hiiiiiiiiiiiiiiii')
    if img_file is not None:
        #img2=img_file.resize((200,100))
        st.image(img_file,use_column_width=False)
        save_image_path = './'+img_file.name
        st.write('HI pranav')
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        if st.button("Predict"):
            ain(save_image_path)
            print(save_image_path)
            print("code succeed")

        
if __name__ == "__main__":
    #[theme]
    #st.base="light"
   
    st.balloons()
    #st.selectbox('Pick one', ['cats', 'dogs'])
    img1 = Image.open(r'C:\Users\Pranav Shinde\Downloads\BE project files\streamlut\butterfly.jpg')
    img1 = img1.resize((350,250))
    st.image(img1,use_column_width=False)
    st.title("Insect Classification ")
    st.markdown('''<h4 style='text-align: middle; color: #8b70e5;font-family: Quando;font-size: 1em;text-transform:capitalize; '>Primates need good nutrition, to begin with. Not only fruits and plants, but insects as well</h4>''',unsafe_allow_html=True)

    activity1 = ["Identification","Information"]
    choice = st.sidebar.selectbox("Select Function",activity1)

    if choice=='Identification':
        t=st.button('Upload file')
        a=st.button('Click picture')
        if (t):
            img_file = st.file_uploader("Upload File or click picture", type=["png","jpg","jpeg"])
            time.sleep(10)
            aod(img_file)
            
            st.success('Done')
            
            
                
    
            
        elif (a):
            
            img_file = st.camera_input('Take a picture')
            st.success('Done')

            if img_file is not None:
                #img2=img_file.resize((200,100))
                st.image(img_file,use_column_width=False)
                save_image_path = './'+img_file.name
                st.write('HI pranav')
                with open(save_image_path, "wb") as f:
                    f.write(img_file.getbuffer())
                    
                if st.button("Predict"):
                    #ain(save_image_path)
                    print(save_image_path)
                    print("code succeed")
                    
    if choice=='Information':
        st.write('Whenever a farmer plants a crop, there are a lot of things that they are counting on to help grow high quality grain. \nFor Greg and his family, they need warm temperatures to help the seed sprout and poke out of the ground. Then they need timely rains, along with plenty of sunshine to help that grain grow. But they also need to make sure that hungry insects or pesky fungal diseases don’t eat away at that crop. The good news is, there are products today that can help prevent some of the pest damage. On today’s tour we talk with Greg about something they use called seed treatments. This coating, that covers each seed, is used to protect it at its most delicate stage of lifeSo come for a tour to find out why Greg thinks this is better than other ways of protecting these little seeds and seedlings.')
    
            
        #st.success("Predicted Bird is: "+result)






    
