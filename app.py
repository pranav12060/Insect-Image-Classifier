# insect classication app by Pranav Shinde.
import streamlit as st
from PIL import Image
def ain(image_name):
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
        
    
if __name__ == "__main__":
 img1 = Image.open('butterfly.jpg')
 img1 = img1.resize((350,350))
 st.image(img1,use_column_width=False)
 st.title("Insect Classification By Pranav Shinde")
 #st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "270 Bird Species also see 70 Sports Dataset"</h4>''',unsafe_allow_html=True)
 img_file = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
 if img_file is not None:
     st.image(img_file,use_column_width=False)
     save_image_path = './upload_images/'+img_file.name
     with open(save_image_path, "wb") as f:
         f.write(img_file.getbuffer())
     if st.button("Predict"):
         ain(save_image_path)
     #st.success("Predicted Bird is: "+result)

    
