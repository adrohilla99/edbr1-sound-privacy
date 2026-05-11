EDBR.1 - Detecting Sounds with Privacy

This project is motivated by a request from a startup company, although this project is entirely separate and is pure academic research.

The idea of this project is to attempt to use the existing ideas behind face-swapping in images/videos, and apply them to try to ensure privacy in sound detection AI.

A good introduction to face-swapping is given in this popular-science article (https://arstechnica.com/science/2019/12/how-i-created-a-deepfake-of-mark-zuckerberg-and-star-treks-data/).  In brief, face-swapping works by training three AI models.  The first looks at a picture of a face and encodes the relevant features of the face (including pose, orientation, etc) into a small amount of data.  The second uses the small amount of data to attempt to reconstruct the original image, when the original image is of person A (say, me).  The third uses the small amount of data to attempt to reconstruct the original image, when the original image is of person B (say, my boss).  The three models are trained together, so that they work well together.  Then, when person A is encoded but the decoder for person B is used, the faces are swapped.

Instead of swapping faces, the idea here is to identify sounds from an audio recording.  For example, to identify "aircraft flying overhead", "birdsong", "loud argument in the street", etc.  However, this detection should happen locally with minimal computational resources, and the only information sent and stored should be anonymous.  E.g. it should record that a loud argument was heard, but not what was said in the argument.  To do this, two AI models would be trained.  The first would listed to an always-on audio feed (say 48kHz 16bit) and, using minimal computational resources, output a much smaller data stream.  The second model would receive the data stream and, using perhaps significantly more computational resources, classify the signal.  However, it should be impossible for the second model to receive too much data --- privacy should be maintained locally.

This project would attempt a proof of concept of this idea.  If time permits (and it may not), it would be interesting to investigate the second model receiving multiple data streams from multiple local audio encoders, with the aim of being able to say, e.g. "that was a loud car driving along this street".

Suggested Type of Project
Not Specified
Suggested Methodological Approach
Not Specified
Data is provided by
student
Supervisor
Ed Brambley
Course
AAI - Applied Artificial Intelligence
Project Tags
AAI - Applied Artificial Intelligence

Supervisor: Ed Brambley
Course: AAI - Applied Artificial Intelligence
