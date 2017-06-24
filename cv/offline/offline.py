class offline_input(object):

    def upload(self, filename):
        """
        upload(filename):
        :param filename: string
        """
        import cv2
        import sys
        import numpy as np
        import os
        
        if not(os.path.isfile(filename)):
            print("file not found")
            sys.exit(1)
        
        output_image = "output.jpg"
        video_capture = cv2.VideoCapture(filename)
        framecount = 0
        count = 0
        success = True

        fps = video_capture.get(cv2.CAP_PROP_FPS)

        while success:
            # Capture frame-by-frame
            success, frame = video_capture.read()
            framecount += 1
            count += 1
            
            # captures frame every 5s
            if int(framecount) % int(fps*5) == 0:
                framecount = 0
                array_image = np.array(frame)
                cv2.imwrite(output_image, frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()