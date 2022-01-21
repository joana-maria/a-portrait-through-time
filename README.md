# A Portrait Through Time 

## Context ##

*A Portrait Through Time* was developed in the context of the Computational Design Laboratory course of the [master’s degree in Design and Multimedia](https://dm.dei.uc.pt/en/) of the Faculty of Sciences and Technology of the [University of Coimbra](https://www.uc.pt/en). We were challenged to explore new ways of developing design artifacts while using computational methods. We had a variety of themes to chose from, and I decided to explore the notion of *Future*.
This project was supervised by Tiago Martins, João M. Cunha, Pedro Silva, and Sérgio M. Rebelo.

## Concept ##

*A Portrait Through Time* represents the idea that the future is the sum of all perspectives of the individual, from the past to the present, alongside its fragmentation and reconstruction.
Death and mortality has been a subject for exploration through different areas such as philosophy, art, literature and music. Art has always been a refugee for humans to explain/express the inexplicable, such as death. *Memento Mori* paintings and *Vanitas* both explore the fragility of life and the fleeting moment.

 ## Objectives ##

Through the overlap of different fractions of the face, the system creates a cubist portrait of an individual where different perspectives and moments are detected and displayed in controlled chaos. The three cameras around the spectator, allows for different panoramas of the face to be captured. Using live transmission, the system represents the past and the present.
The portrait intends to reference the three time frames live but also immor- talize the moment with an input. After the Space Bar is pressed a timer goes of and a static visualization of the portrait is created. This interaction immortalizes the portrait with pausing all movement in all units and adds another dimension to the future section of the portrait.
This frame from the source video suffers a style transfer operation with an image from a Vanitas painting or a Memento Mori painting selected randomly from six curated images


 ## Running Project ## 

This project uses [Yarn](https://yarnpkg.com/en/) for dependencies.

**Step 1**

To run it locally, you must install Yarn and run the following command at the repository's root to get all the dependencies.

`yarn run prep`

**Step 2**

Then, you can run 

`yarn run start`

**Step 3**

This project was create to run with three different video inputs.
In order to use the repository, you have to change the id’s of those devices. After running you’re able to see on the console which are the correspondent id’s of the video devices in your computer. Chose and update the variables `id_front_cam` , `id_left_cam`, `id_right_cam` on the `main.js` page to the id’s that appear on the console.

Note: That if you want to only use one camera you can 😉. Just update all the variables to the id of your video devices.

**Step 4**

Press the Space Bar, and POSE ✨
Wait a few seconds, and let the system take a static version of your portrait, .
Watch the style tranfer being applied. To go back you just need to press the space bar again. 

## Credits and References ##

Based on:

- [https://github.com/reiinakano/arbitrary-image-stylization-tfjs](https://github.com/reiinakano/arbitrary-image-stylization-tfjs) - Used to create the style transfer.
- [https://ml5js.github.io/ml5-examples/javascript/FaceApi/FaceApi_Video_Landmarks/](https://ml5js.github.io/ml5-examples/javascript/FaceApi/FaceApi_Video_Landmarks/) - Used to detect different parts of the face.

## Have any doubts? ##

In case you have any doubts about the project please e-mail jmoliveira@student.dei.uc.pt.
