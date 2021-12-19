# insect classication app by Pranav Shinde.
import streamlit as st
def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        plt.imshow(image)


if __name__ == "__main__":
    main()

