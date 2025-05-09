# Nano_Drone_For_Indoor_Crime_Scene_Analysis
 
Goodies to help you to work with crime scene evidence with a drone

Basic Concept

Crime is a horrible problem, and investigations are often hindered by various factors such as human contamination and potentially inefficient or inaccurate manual methods. We believe drones can be used to rapidly access a crime scene to provide a first response, mapping and gathering evidence, and analyzing. 
(Details in an exploratory paper/video for ARSO 2025, which has been accepted Apr. 29, 2025: https://www.youtube.com/watch?v=An8TIGr_NRA) 
This rep contains for now just some code for estimating direction a bloody object was moving from bloodstain traces, like a criminal moving with bloody shoes, or a bloody corpse being dragged. (Note: these are called transfer bloodstains, smears, or swipes.) Later some extra code might or might not be added...

Content (requirements, files)

Requirements: environment with Python 3 and OpenCV

Folders: blood_smear_direction: just run the python program to see how this works, e.g., with "python detect_direction_bloody_object_moved.py" in your environment.
This processes 20 data samples from input/raw_images and puts the results in output.

Basically, as the paper says, we created five samples with red dye on white paper:
Sample 1 contains handprints from a person (such as a "victim") crawling with a "bloody" hand.
Sample 2 contains shoeprints from a person (such as a "criminal") walking away with "blood" on their shoe.
Sample 3 contains a dragged handprint (like a body being dragged where the palm of a "bloody" hand is touching the floor)
Sample 4 contains amorphous stains from a wet tissue (e.g., like someone crawling while touching the floor with some "bloodied" clothed part of their body, like a knee)
Sample 5 contains a dragged amorphous stain from a wet tissue (e.g. like a body being dragged with some "bloodied"/clothed part of their body touching the floor).

Then, a drone was used to gather images over the five samples we created to simulate crime scene analysis.
For each sample, we extracted four images, where the bloodsmear is aligned such that the direction of motion is up, left, down, or right, yielding a total of 20 images.
The code uses a simplified approach involving color picking and image moments to estimate direction, and provides feedback on accuracy and average error, etc.

(Note: This code was written using the author's setup described above for exploratory research purposes; the author cannot help with getting it to work on the reader's system.)
