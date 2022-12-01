from typing import Optional
from pupil_apriltags import Detector
import cv2

COMPETITION_FAMILY = "tag16h5"
FILTER_MATCHES = 60

class DetectorSettings:
    def __init__(self, threads, quad_decimate, quad_sigma, refine_edges, decode_sharpening):
        self.threads = threads
        self.quad_decimate = quad_decimate
        self.quad_sigma = quad_sigma
        self.refine_edges = refine_edges
        self.decode_sharpening = decode_sharpening
        
    def __str__(self):
        return \
f"""DetectorSettings {{ 
    threads = {self.threads}, 
    quad {{
        decimate = {self.quad_decimate},
        sigma = {self.quad_sigma}
    }},
    refine_edges = {self.refine_edges}, 
    decode_sharpening = {self.decode_sharpening}
}}"""

class AprilTagDetector:
    def __init__(self, settings: DetectorSettings, family = COMPETITION_FAMILY):
        self.__detector = Detector(
            families = family,
            nthreads = settings.threads,
            quad_sigma        = settings.quad_sigma,
            quad_decimate     = settings.quad_decimate,
            refine_edges      = settings.refine_edges,
            decode_sharpening = settings.decode_sharpening
        )
    
    def id_from(self, img) -> Optional[int]:
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        response = self.__detector.detect(grayed)

        # not sure what we need to do with the tags yet, I'm just returning the ID.
        if len(response) > 0 and response[0].decision_margin > FILTER_MATCHES:
            return response[0]
        
        return None

"""
Default export, this is the AprilTagDetector we will be using.
"""
default_detector = AprilTagDetector(settings = DetectorSettings(
    threads = 1,
    quad_decimate = 0.0,
    quad_sigma = 1.0,
    refine_edges = 1,
    decode_sharpening = 0.25
))
