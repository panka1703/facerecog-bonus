import sys
import argparse
import cv2
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from time import time
from libfaceid.liveness    import FaceLivenessModels, FaceLiveness

# Use flask for web app
from flask import Flask, render_template, Response
app = Flask(__name__)

# Set the window name
WINDOW_NAME = "Facial_Recognition"

# Set the input directories
INPUT_DIR_DATASET               = "datasets"
INPUT_DIR_MODEL_DETECTION       = "models/detection/"
INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
INPUT_DIR_MODEL_TRAINING        = "models/training/"
INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"
INPUT_DIR_MODEL_LIVENESS        = "models/liveness/"
# Set width and height
RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)
RESOLUTION_HD     = (1280, 720)
RESOLUTION_FULLHD = (1920, 1080)



def cam_init(cam_index, width, height): 
    cap = cv2.VideoCapture(cam_index)
    if sys.version_info < (3, 0):
        cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        if confidence is not None:
            text = "{} {:.2f}%".format(face_id, confidence)
        else:
            text = "{}".format(face_id)
        #cv2.putText(frame, "{} {:.2f}%".format(face_id, confidence),         
        #    (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

##edit
def monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close):
    if eyes_close:
        #print("eye less than threshold {:.2f}".format(eyes_ratio))
        eye_counter += 1
    else:
        #print("eye:{:.2f} blinks:{}".format(eyes_ratio, total_eye_blinks))
        if eye_counter >= eye_continuous_close:
            total_eye_blinks += 1
        eye_counter = 0
    return total_eye_blinks, eye_counter
##edit
def monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open):
    if mouth_open:
        #print("mouth more than threshold {:.2f}".format(mouth_ratio))
        mouth_counter += 1
    else:
        #print("mouth:{:.2f} opens:{}".format(mouth_ratio, total_mouth_opens))
        if mouth_counter >= mouth_continuous_open:
            total_mouth_opens += 1
        mouth_counter = 0
    return total_mouth_opens, mouth_counter
def process_livenessdetection():

    cam_index = 0
    cam_resolution = RESOLUTION_QVGA
#    model_detector=FaceDetectorModels.HAARCASCADE
    model_detector=FaceDetectorModels.DLIBHOG
#    model_detector=FaceDetectorModels.DLIBCNN
#    model_detector=FaceDetectorModels.SSDRESNET
#    model_detector=FaceDetectorModels.MTCNN
#    model_detector=FaceDetectorModels.FACENET

#    model_recognizer=FaceEncoderModels.LBPH
#    model_recognizer=FaceEncoderModels.OPENFACE
    model_recognizer=FaceEncoderModels.DLIBRESNET
#    model_recognizer=FaceEncoderModels.FACENET
    #liveness=FaceLivenessModels.EYESBLINK_MOUTHOPEN
    liveness=FaceLivenessModels.COLORSPACE_YCRCBLUV

    # Initialize the camera
    camera = cam_init(cam_index, cam_resolution[0], cam_resolution[1])
    check=int(input("Enter number 1 if you want to show live video and vice versa: "))
    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
        # Initialize face recognizer
        face_encoder = FaceEncoder(model=model_recognizer, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
        # Initialize face liveness detection
        face_liveness  = FaceLiveness(model=FaceLivenessModels.EYESBLINK_MOUTHOPEN, path=INPUT_DIR_MODEL_LIVENESS)
        face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)

    except:
        #face_encoder = None
        print("Warning, check if models and trained dataset models exists!")
        return
    face_id, confidence = (None, 0)

    ##edit
    eyes_close, eyes_ratio = (False, 0)
    total_eye_blinks, eye_counter, eye_continuous_close = (0, 0, 1) # eye_continuous_close should depend on frame rate
    mouth_open, mouth_ratio = (False, 0)
    total_mouth_opens, mouth_counter, mouth_continuous_open = (0, 0, 1) # eye_continuous_close should depend on frame rate

    time_start = time()
    time_elapsed = 0
    frame_count = 0
    identified_unique_faces = {} # dictionary
    runtime = 1000 # monitor for 10 seconds only
    is_fake_count_print = 0
    is_fake_count_replay = 0
    face_count=0
    ##edit
    time_recognition=5 
    checkface=False


    print("Note: this will run for {} seconds only".format(runtime))

    while (time_elapsed < runtime):

        # Capture frame from webcam
        ret, frame = camera.read()
        if frame is None:
            print("Error, check if camera is connected!")
            break


        ## Detect and identify faces in the frame
        #faces = face_detector.detect(frame)
        #for (index, face) in enumerate(faces):
            #(x, y, w, h) = face
            ## Indentify face based on trained dataset (note: should run facial_recognition_training.py)
            #if face_encoder is not None:
                #face_id, confidence = face_encoder.identify(frame, (x, y, w, h))
            ## Set text and bounding box on face
            #label_face(frame, (x, y, w, h), face_id, confidence)

            # Process 1 face only
            #break
        # Detect and identify faces in the frame
        # Indentify face based on trained dataset (note: should run facial_recognition_training.py)
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):

            # Check if eyes are close and if mouth is open
            eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face)
            mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face)
            #print("eyes_close={}, eyes_ratio ={:.2f}".format(mouth_open, mouth_ratio))
            #print("mouth_open={}, mouth_ratio={:.2f}".format(mouth_open, mouth_ratio))

            # Detect if frame is a print attack or replay attack based on colorspace
            is_fake_print  = face_liveness2.is_fake(frame, face)
            is_fake_replay = face_liveness2.is_fake(frame, face, flag=1)

            # Identify face only if it is not fake and eyes are open and mouth is close
            if is_fake_print:
                is_fake_count_print += 1
                face_id, confidence = ("Fake", None)
            elif is_fake_replay:
                is_fake_count_replay +=1
                face_id,confidence = ("Fake", None)
            elif not eyes_close and not mouth_open:
                face_id, confidence = face_encoder.identify(frame, face)
                if (face_id not in identified_unique_faces) & (confidence >50) :
                    identified_unique_faces[face_id] = 1
                elif (face_id in identified_unique_faces) & (confidence >50):
                    identified_unique_faces[face_id] += 1
            
            if (face_count>100) | (face_id=="Fake"):
                face_count=0
            elif (face_id !="Fake") & (confidence > 50):
                face_count+=1
               
            print("Identifying: {:.2f} %".format((face_count/100)*100))
            

            label_face(frame, face, face_id, confidence) # Set text and bounding box on face
            break # Process 1 face only


        # Monitor eye blinking and mouth opening for liveness detection
        total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close)
        total_mouth_opens, mouth_counter = monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open)


        # Update frame count
        frame_count += 1
        time_elapsed = time()-time_start
        
            #cv2.imshow(WINDOW_NAME, frame)
        if face_count>99:
            checkface=True
            break
        else:
            checkface=False
    

        # Display updated frame to web app
        if check==1:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')
    if checkface==True:
        print("Hello {}!!!".format(max(identified_unique_faces, key=identified_unique_faces.get)))
    else:
        print("Can not indentify your face, please try again!")
    # Release the camera
    camera.release()
    cv2.destroyAllWindows()
##edit
def run(cam_index, cam_resolution):
    #detector=FaceDetectorModels.HAARCASCADE
    detector=FaceDetectorModels.DLIBHOG
#    detector=FaceDetectorModels.DLIBCNN
#    detector=FaceDetectorModels.SSDRESNET
#    detector=FaceDetectorModels.MTCNN
#    detector=FaceDetectorModels.FACENET

#    encoder=FaceEncoderModels.LBPH
#    encoder=FaceEncoderModels.OPENFACE
    encoder=FaceEncoderModels.DLIBRESNET
#    encoder=FaceEncoderModels.FACENET

#    liveness=FaceLivenessModels.EYESBLINK_MOUTHOPEN
    liveness=FaceLivenessModels.COLORSPACE_YCRCBLUV

    process_livenessdetection(detector, encoder, liveness, cam_index, cam_resolution)
##edit
def main(args):
    if sys.version_info < (3, 0):
        print("Error: Python2 is slow. Use Python3 for max performance.")
        return

    cam_index = int(args.webcam)
    resolutions = [ RESOLUTION_QVGA, RESOLUTION_VGA, RESOLUTION_HD, RESOLUTION_FULLHD ]
    try:
        cam_resolution = resolutions[int(args.resolution)]
    except:
        cam_resolution = RESOLUTION_QVGA

    if args.detector and args.encoder and args.liveness:
        try:
            detector = FaceDetectorModels(int(args.detector))
            encoder  = FaceEncoderModels(int(args.encoder))
            liveness = FaceLivenessModels(int(args.liveness))
            print( "Parameters: {} {} {}".format(detector, encoder, liveness) )
            process_livenessdetection(detector, encoder, liveness, cam_index, cam_resolution)
        except:
            print( "Invalid parameter" )
        return
    run(cam_index, cam_resolution)
##edit
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=False, default=0, 
        help='Detector model to use. Options: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET')
    parser.add_argument('--encoder', required=False, default=0, 
        help='Encoder model to use. Options: 0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET')
    parser.add_argument('--liveness', required=False, default=0, 
        help='Liveness detection model to use. Options: 0-EYESBLINK_MOUTHOPEN, 1-COLORSPACE_YCRCBLUV')
    parser.add_argument('--webcam', required=False, default=0, 
        help='Camera index to use. Default is 0. Assume only 1 camera connected.)')
    parser.add_argument('--resolution', required=False, default=0,
        help='Camera resolution to use. Default is 0. Options: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD')
    return parser.parse_args(argv)
# Initialize for web app
@app.route('/')
def index():
    return render_template('web_app_flask.html')

# Entry point for web app
@app.route('/video_viewer')
def video_viewer():
    return Response(process_livenessdetection(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("\n\nNote: Open browser and type http://127.0.0.1:5000/ or http://ip_address:5000/ \n\n")
    # Run flask for web app
    main(parse_arguments(sys.argv[1:]))
    app.run(host='0.0.0.0', threaded=True, debug=True)
    
