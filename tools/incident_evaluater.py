import math



class Evaluate_Incidents:
    def __init__(self, classes, colors=None):
        self.classes = classes
        self.colors = colors
        self.objects = {}
        self.TTL = 240  # Number of frames before a track is removed
        self.PF = 2#7  # PF = Previous Frame: Number of frames used to determine direction
        self.STOPPED_DISTANCE = 3  # Distance in number of pixels from current to previous frame to determine stopped vehicle
        self.min_number_of_frames = 2#24  # How many frames must there be to evaluate stopped vehicle
        self.update_number_of_frames = 2#12  # How often stopped vehicle should be evaluated
    
    @property
    def colors(self):
        return self._colors
    
    @colors.setter
    def colors(self, colors):
        colors_default = {"alarm": (255,128,128), "ok": (128,128,255)}
        if colors and colors.get("alarm") and colors.get("ok"):
            colors_default = colors
        self._colors = colors_default

    def purge(self, frame_number):
        if frame_number % 24 != 0:
            return
        dict_of_objects = self.objects.copy()
        for object in dict_of_objects:
            if dict_of_objects[object]["last_frame"] < frame_number - self.TTL:
                del self.objects[object]

    def simple_linear_regression(self, track_id, frame_number):
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= 5:
            return None, None

        current_point = (int(track["center_points"][-1][0]), int(track["center_points"][-1][1]))
        if frame_number % 12 != 0:
            direction = self.objects[track_id].get("direction")
            if direction:
                next_point_x = current_point[0] + direction["distance"]
                next_point_y = direction["alpha"] + direction["beta"] * next_point_x
                next_point = (int(next_point_x), int(next_point_y))
                
                return current_point, next_point
            #else:
                #return None, None
        
        if n > 10:
            n = 10
        center_points = track["center_points"][-n:]
        
        x_sum = 0
        y_sum = 0
        for center_point in center_points:
            x_sum += center_point[0]
            y_sum += center_point[1]
            
        x_mean = x_sum / n
        y_mean = y_sum / n

        numerator = 0
        denominator = 0
        for center_point in center_points:
            x = center_point[0]
            y = center_point[1]
            numerator = (x - x_mean) * (y - y_mean)
            denominator = (x - x_mean) ** 2
        
        try:
            beta = numerator / denominator
        except Exception as e:
            print(e)
            beta = 0
        
        alpha = y_mean - beta * x_mean
        
        """
        l = 5
        a = (1-beta**2)
        b = 2 * (alpha * beta - beta * current_point[1] - current_point[0])
        c = current_point[0]**2 + current_point[1]**2 + alpha**2 - 2 * alpha * current_point[1] - l**2

        try:
            x0 = (-b + math.sqrt(b**2 - 4*a*c)) / 2*a 
        except Exception as e:
            print(e)
            x0 = None
        try:
            x1 = (-b - math.sqrt(b**2 - 4*a*c)) / 2*a 
        except Exception as e:
            print(e)
            x1 = None
        
        if not x0 and not x1:
            return None, None
        
        print(f"x0 = {x0}, x1 = {x1}")
        if not x0:
            next_point_x = x0
        else:
            next_point_x = x1

        next_point_y = alpha + beta * next_point_x
        next_point = (int(next_point_x), int(next_point_y))

        """
        d = 1
        if (center_points[-1][0] - center_points[-2][0]) < 0:
            d = -1
        #next_point_x = center_points[-1][0] + 50 * d
        distance = d * math.sqrt((center_points[-1][0] - center_points[-2][0])**2 + (center_points[-1][1] - center_points[-2][1])**2)
        next_point_x = center_points[-1][0] + distance
        next_point_y = alpha + beta * next_point_x
        next_point = (int(next_point_x), int(next_point_y))
        
        self.objects[track_id]["direction"] = {"alpha": alpha, "beta": beta, "distance": distance}
        return current_point, next_point

    def simple_direction(self, track_id, frame_number):
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= 8:
            return None, None

        current_point = (int(track["center_points"][-1][0]), int(track["center_points"][-1][1]))
        if frame_number % 12 != 0:
            direction = self.objects[track_id].get("direction")
            if direction:
                x_vector = direction["x_vector"]
                y_vector = direction["y_vector"]
                length = direction["length"]
                next_point = (int(current_point[0] + x_vector * length), int(current_point[1] + y_vector * length))
                
                return current_point, next_point
        
        previous_point = track["center_points"][-self.PF]

        x_vector = current_point[0] - previous_point[0]
        y_vector = current_point[1] - previous_point[1]
        length_vector = math.sqrt(x_vector**2 + y_vector**2)
        try:
            x_vector /= length_vector
            y_vector /= length_vector
        except Exception as e:
            print(e)
            return None, None

        length= 50

        next_point = (int(current_point[0] + x_vector * length), int(current_point[1] + y_vector * length))

        self.objects[track_id]["direction"] = {"length": length, "x_vector": x_vector, "y_vector": y_vector}
        return current_point, next_point

    def pedestrian(self, class_name):
        if class_name == "person":
            return True
            #color = self.colors["alarm"]
            #text = "INCIDENT: Pedestrian"
        #else:
        #    color = self.colors["ok"]
        return False
    
    def stopped_vehicle(self, track_id, frame_number):
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= self.min_number_of_frames:
            return False

        current_point = (int(track["center_points"][-1][0]), int(track["center_points"][-1][1]))
        if frame_number % self.update_number_of_frames != 0:
            stopped = self.objects[track_id].get("stopped")
            if stopped:
                return True
            return False
        
        previous_point = track["center_points"][-self.PF]

        distance = math.sqrt((current_point[0] - previous_point[0])**2 + (current_point[1] - previous_point[1])**2)

        if distance <= self.STOPPED_DISTANCE:
            self.objects[track_id]["stopped"] = True
            return True
        self.objects[track_id]["stopped"] = False
        return False
    
    def opposite_driving(self, track_id, frame_number):
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= 24:
            return False

        current_point = (int(track["center_points"][-1][0]), int(track["center_points"][-1][1]))
        if frame_number % 24 != 0:
            stopped = self.objects[track_id].get("stopped")
            if stopped:
                return False
            return False
        
        previous_point = track["center_points"][-self.PF]

        distance = math.sqrt((current_point[0] - previous_point[0])**2 + (current_point[1] - previous_point[1])**2)

        if distance <= self.STOPPED_DISTANCE:
            self.objects[track_id]["stopped"] = False
            return False
        return False

    def evaluate(self, track, frame_number):
        class_name = track.get_class()
        #print(f"{class_name} - {track.track_id}")
        text = f"{class_name} - {track.track_id}"
        color = self.colors["ok"]
        bbox = track.to_tlbr()
        center_point = ((int(bbox[0]) + (int(bbox[2]) - int(bbox[0])) / 2), int(bbox[1]) + (int(bbox[3]) - int(bbox[1])) / 2)
        if track.track_id in self.objects:
            self.objects[track.track_id]["center_points"].append(center_point)
            self.objects[track.track_id]["last_frame"] = frame_number
        else:
            self.objects[track.track_id] = {"center_points": [center_point], "last_frame": frame_number}
        
        #current_point, next_point = self.simple_linear_regression(track.track_id, frame_number)
        current_point, next_point = self.simple_direction(track.track_id, frame_number)

        if self.pedestrian(class_name):
            color = self.colors["alarm"]
            text = "INCIDENT: Pedestrian"
            current_point, next_point = None, None
        elif self.stopped_vehicle(track.track_id, frame_number):
            color = self.colors["alarm"]
            text = "INCIDENT: Stopped vehicle"
            current_point, next_point = None, None
        #elif self.opposite_driving(track.track_id, frame_number):
        #    color = self.colors["alarm"]
        #    text = "INCIDENT: Stopped vehicle"
        
        return color, text, current_point, next_point
        
