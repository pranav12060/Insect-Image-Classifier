import cv2
import numpy as np 
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib



import streamlit as st
import time
from PIL import Image
import numpy as np


kmeans, scale, svm, im_features =joblib.load("bovw.pkl")
def getDescriptors(sift, img):
    
    kp, des = sift.detectAndCompute(img, None)
    return des
def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img,(150,150))
def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 
    return descriptors
def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1,32)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    return im_features
def normalizeFeatures(scale, features):
    return scale.transform(features)
def testModel(path, kmeans, scale, svm, im_features, no_clusters, kernel):
    count = 0
    true = []
    descriptor_list = []

    name_dict =	{
        "0": "Aphid",
        "1": "caterpillar",
        "2":"corn flea beetle",
        "3":"Red Spider Mites",
        "4":"whitefly"
      
    }

    sift = cv2.ORB_create()

    img_path=path
    img = readImage(img_path)
    des = getDescriptors(sift, img)
    if(des is not None):
        count += 1
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)
    test_features = scale.transform(test_features)    
    kernel_test = test_features
    if(kernel == "precomputed"):
        kernel_test = np.dot(test_features, im_features.T)
   
    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    #print("Given picture is of:")
    #print(predictions)
    return predictions


#Info about insect
def iai(name):
    
    with st.expander("More Information about"):
        
        if(name=="Red Spider Mites"):
            
            st.write("""
                        What is a Red Spider Mite?\n
                        Red spider mites can be one of two kinds of mites, either the European red spider mite or the Southern red spider mite.
                        The most common red spider mite is the Southern variety. The European spider mite is normally only seen on apple trees,
                        while the Southern spider mite attacks a much wider variety of plants.\n
                        Identifying Red Spider Mites\n
                       A plant that is infested by red spider mites will start to look unhealthy and will have a dusty appearance on the undersides of its leaves.
                       Close inspection will reveal that the dust is actually moving and is in fact the spider mites. The plant may also have some webbing on the
                       underside or on the branches of the plant. You cannot easily make out the details of red spider mites with the naked eye but a simple magnifying
                       glass can make the details more visible. A red spider mite will be all red. There are other kinds of spider mites, such as the two-spotted spider
                       mite, that are partially red. Red spider mites will be all red. Knocking some off onto a piece of white paper will make it easier to distinguish the colors.
                        \n
                     How to Control Red Spider Mites \n
                     Red spider mites are most active in cool weather, so you are most likely to see an infestation of them in the spring or fall. The best way to control red spider mites is
                     through the use of their natural predators. Lacewings and ladybugs are commonly used, but predatory mites can also be used. All of these spider mite predators are
                     available from reputable gardening supply centers and websites. You can also use pesticides to eliminate red spider mites. Insecticidal soaps and oils work best. You
                     should be careful using pesticides though as they will also kill their natural predators and the red spider mites may simply move from the pesticide-treated area to
                     non-treated areas. Of course, the best way to eliminate red spider mites is to make sure you don’t get them in the first place. Work to keep plants healthy and the
                     areas around the plants free of debris and dust to keep red spider mites away. Also, make sure plants have enough water. The water will help keep the red spider mites
                     away as they prefer very dry environments.
                     """)
            st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.pLYsybUVzgoVkz6VTXHyBAHaFj%26pid%3DApi&f=1")

        elif(name=="whitefly"):
            st.write("""
                        What Are Whiteflies?\n
                        Whiteflies are soft-bodied, winged insects closely related to aphids and mealybugs. Despite their name, whiteflies are not a type of fly, though
                        they do have wings and are capable of flying.Whiteflies can be as small as 1/12 of an inch, are somewhat triangular in shape, and are often
                        found in clusters on the undersides of leaves. They are active during the day and will scatter when disturbed, so they can be easier to spot
                        than some nocturnal insect pests.There are hundreds of species of whiteflies, but most affect only a small number of host plants. However,
                        there are a few whitefly species that affect a wider range of plants, which make them the most problematic in horticulture. These whitefly
                        species include the greenhouse whitefly, bandedwinged whitefly, giant whitefly, and silverleaf whitefly, among others. Silverleaf whiteflies,
                        which are slightly smaller and more yellow than other whiteflies, are especially common in the southern United States
                        \n
                        How to Identify Whiteflies\n
                        Like aphids, whiteflies use their piercing mouthparts to suck up plant juices and, in turn, produce a sticky substance known as honeydew.
                        Honeydew left on its own can cause fungal diseases such as sooty mold to form on leaves. With heavy whitefly feeding, plants will quickly
                        become extremely weak and may be unable to carry out photosynthesis. Leaves will wilt, turn pale or yellow, growth will be stunted, and
                        eventually leaves may shrivel and drop off the plant.Honeydew is a sign that the whiteflies have been feeding for several days. You might
                        also see ants, which are attracted to the sweet honeydew.
                        \n
                        How to Control Whiteflies\n
                        Your first line of defense should be inspecting all plants for pests before you bring them home, as well as keeping any new additions away
                        from the rest of your plants for a period of time. This will allow you to identify and curtail any pest or disease issues that appear.
                        Keeping natural predators around will prevent whiteflies from ever exploding in population. For this reason, avoid using insecticides.
                        Ladybugs, spiders, green lacewing larvae, and dragonflies are a few of many beneficial insects that can control a whitefly population.
                        Hummingbirds are another natural predator. Try creating a habitat that will attract dragonflies and damselflies (which also helpfully
                        eat mosquitoes) or beautiful hummingbirds. When it comes to whiteflies, avoid chemical insecticides; they’re usually resistant and all
                        you end up doing is killing the beneficial insects—their natural predators—and the insects that pollinate the garden for a better harvest!
                        Mulch early in the season with aluminum reflective mulch, especially around tomatoes and peppers. The reflective mulch makes it challenging
                        for whiteflies to find their preferred host plants. Set out yellow index cards coated with petroleum jelly to monitor whiteflies,
                        especially when it comes to tomatoes, peppers, sweet potatoes, or cabbage crops. A half-and-half mixture of petroleum
                        jelly and dish soap, spread over small boards painted bright yellow, is sticky enough to catch little whiteflies, too. To whiteflies,
                        the color yellow looks like a mass of new foliage. The bugs are attracted to the cards, get stuck in the jelly, and die.
                       """)
            st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.SB8IkL1h4HRrYejTveXJwAHaE7%26pid%3DApi&f=1")

        elif(name=="Aphid"):
            st.write("""
                        What Are Aphid?
                        \nAphids are very common insects and are found on most plants in yards and gardens. In most cases they cause little or no damage to the health
                        of plants. Signs of severe aphid feeding are twisted and curled leaves, yellowed leaves, stunted or dead shoots and poor plant growth.
                        Treating aphids for the health of plants is usually unnecessary. Aphids can often be managed with only non-chemical options or low risk pesticides.
                        \nHow to identify aphids
                        \nAphids are small, 1/16- to 1/8-inch-long (2-4 mm), pear-shaped, soft-bodied insects. They can range in color from green, black, red,
                        yellow, brown or gray. Mature aphids can be wingless or can have wings.Winged aphids are similar in color but are a little darker.
                        Immature aphids (nymphs) look like adults  but are smaller. 
                        The best way to identify aphids is to check for two tail pipes (cornicles) found at the end of the abdomen. All aphids have cornicles,
                        but some are smaller and less obvious.
                        \nHow to Control Aphids
                        \nWater: Spray aphids off of plants with a strong stream of water from a garden hose. This method is most effective early on in the season
                        before an infestation has  fully taken hold. It may not be a good choice for younger or more delicate plants, but it works well on plants
                        where you can use higher water pressure.\n Remove by hand: Put on some garden gloves and knock them off of stems, leaves, flower buds,
                        or wherever you see them, and into a buck et of soapy water to kill them.You can also cut or prune off the affected areas and drop
                        them into the bucket.
                        \nSoap and water: Make a homemade aphid spray by mixing a few tablespoons of a pure liquid soap (such as castile) in a small bucket
                        of water. (Avoid using detergents or products with degreasers or moisturizers.) Apply with a spray bottle directly on aphids
                        and the affected parts of the plant, making sure to soak the undersides of leaves where eggs and larvae like to hide. The soap
                        dissolves the protective outer layer of aphids and other soft-bodied insects, eventually killing them. It doesn’t
                        harm birds or hard-bodied beneficial insects like lacewings, ladybugs or pollinating bees. You can also purchase ready-to-use
                        insecticidal soaps online or at a local nursery.\noil: The organic compounds in neem oil act as a repellent for aphids and other insects,
                        including mealybugs, cabbage worms,
                        beetles, leafminers, ants and various types of caterpillars. However, it may repel beneficial insects, so use caution when and
                        where they are present. Follow package instructions for diluting the oil in water or use a ready-to-use neem oil spray, and
                        spray the affected areas. Neem oil is also good for controlling different types of fungus.\nEssential oils: Create your own spray mixture
                        with essential oils. Use 4 to 5 drops of each: peppermint, clove, rosemary and
                        thyme, and mix with water in a small spray bottle. Spray on affected plants to target adult aphids, as well as aphid larvae and eggs.
                     """)
            st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.hC1wjljo6kSyvOKtEcn75AHaE0%26pid%3DApi&f=1")
            
        elif(name=="corn flea beetle"):
            st.write("""
                     What are corn flea beetle
                     \nFlea beetles are common pests found on many vegetable crops including radishes, broccoli, cabbage, turnips, eggplant, peppers, tomatoes, potatoes, spinach and melons.
                     Flea beetles chew irregular holes in the leaves. Severe flea beetle damage can result in wilted or stunted plants. Flea beetles are best managed through a combination
                     of cultural and chemical control methods.
                     
                     \nHow to identify flea beetles
                     \nMost adult flea beetles are very small (1/16 –1/8 inch long). An exception is the spinach flea beetle, which is 1/4-inch long. Flea beetles can be black, bronze,
                     bluish or brown to metallic gray. Some species have stripes. All flea beetles have large back legs which they use for jumping, especially when disturbed.
                     
                     \nHow to control corn flea beetle
                    \nOrganic spray: Whether it’s one you create yourself or purchase from a garden supply store, sprays are reasonably effective against these annoying bugs.
                    Spray the plant, and as the beetles eat the contaminated foliage, they’ll slowly be poisoned. It can take a few days before you notice the beetles dying off.
                    To ensure the pest is eradicated, spray a few times during the gardening season.
                    \nNeem oil spray: Neem oil applied to the foliage of many garden plants keeps away a host of pests, including flea beetles. It needs to be re-applied after rain
                    and usually needs to be applied multiple times before it does the job.
                    \nAs with nearly every garden problem, prevention is a lot easier than control. If you want to avoid flea beetle damage altogether, here’s how to do it.
                    First, as a general rule, check on your garden plants often. The more often you head out to the garden to inspect your plants, the quicker you’ll spot problems.
                    Ignoring damage for too long means that the pests are more likely to have already transmitted a disease. Don’t forget to take notes about what’s going on in your garden. Did you spot an infestation this year? Knowing when you transplanted and covered crops can help you determine better strategies for next year.
                    """)
            st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.1A5QyuMqXeK8b29YlmjPMgEsEs%26pid%3DApi&f=1")

        elif(name=="caterpillar"):
            st.write("""
                        What Are Caterpillars?
                        \nCaterpillars are the larval stage of the class of insects called Lepidoptera, commonly known as butterflies and moths. They spend their days eating and storing energy to become adult
                        butterflies or moths. Caterpillars are well-adapted to their natural surroundings. Most of them are camouflaged, so even though they're all around us, we usually don't see them.
                        They are so perfectly disguised (or have such secretive habits) that we walk right by them without ever knowing they're there. But they are! Butterflies and moths go through
                        "complete metamorphosis" —that is, they have four distinct stages: egg, larva, pupa, adult. The caterpillar is the larval stage, and all a caterpillar does is eat and store energy
                        for the adult stage. They are basically eating machines whose only goal is to store fat for adulthood. Caterpillars are cool! They are often camouflaged, but many have bright colors
                        and patterns that may serve to warn or scare away predators, like birds. Most caterpillars are totally defenseless, but a few species are protected by stinging spines.

                        \nHow to control caterpillar
                        \nControl is not always necessary for caterpillars, which may only feed for a short time before pupating into moths and butterflies. When control is required, hand-picking
                        is among the most effective solutions. Each day, go to the garden while the troublesome caterpillars are feeding to pick them from plants and drop them into a bucket of
                        soapy water. Even when caterpillars are no longer found, you should check the plants weekly for newly emerged larvae. Sometimes, borers or leafrollers developing in tight concentrations can be cut from affected plants, but borers inside the trunks of trees are extremely difficult to treat without professional help.
                        Bacillus thuringiensis (Bt) is the most effective commercially available spray for caterpillars, but must be ingested by the caterpillar before it can work. You can spray Bt mixed at a ratio of 1/2 to 4 teaspoons per gallon of water on any plant where hand-picking and pruning aren't viable options. It has a short
                        life when exposed to sunlight, and it should be reapplied twice weekly until the caterpillars are gone. Bt is safe for bees and predatory insects.
                        """)
            st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.AUHg2Wyc7BmwvglPzzmePwHaE8%26pid%3DApi&f=1")





def run():
  
    st.markdown('''
                <img src="http://yesofcorsa.com/wp-content/uploads/2017/02/Butterfly-Wallpaper-For-Desktop-1024x683.jpg" alt="Paris"
                style=" border: 1px solid #ddd;
                          border-radius: 43px;
                          padding: 2.5px;
                          width:400px">
                    ''',
                    unsafe_allow_html=True  
    )
    
 
    st.title("Insect Classification")
    st.markdown('''<h4 style='text-align: middle; color: #8b70e5;font-family: Quando;font-size: 1em;text-transform:capitalize; '>Primates need good nutrition, to begin with.
                Not only fruits and plants, but insects as well</h4>''',unsafe_allow_html=True)
    
    img_file = st.file_uploader("Choose an Image of Insect", type=["jpg", "png"])
   
    if img_file is not None:
        st.write('uploaded image')
        st.image(img_file,use_column_width=False,width=350)
        save_image_path = './images_uploaded_on_heroku/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = testModel(save_image_path, kmeans, scale, svm, im_features, 10, "precomputed")
            st.success("Given insect is: "+result[0])
            iai(result[0])  #info about insect
            
            

#code starts from here            
if __name__ == "__main__":

    # setting page 
    st.set_page_config(
   page_title="InsectSpot",
   page_icon="🦋",
   #page_icon="🧊"  #layout="wide",
   initial_sidebar_state="expanded",
)
    #method to execute
    run()
   
