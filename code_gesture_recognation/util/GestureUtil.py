class HandModel:

    def get_hand_position(self):
        pass

    def get_model_parameter(self):
        pass

    def get_hand_classification(self):
        pass

    def get_hand_name(self):
        pass


class HandSkeletalModel(HandModel):

    def __init__(self, hand_lm, property_hand):
        self.hand_lm = hand_lm
        lista = str(property_hand).split()
        self.property_hand = HandProperty(lista[lista.index('score:') + 1], lista[lista.index('index:') + 1],
                                          lista[lista.index('label:') + 1])
        x = 0
        y = 0
        for point in self.hand_lm:
            x += int((point.x/point.denormalization_x)*point.denormalization_y)
            y += int((point.y/point.denormalization_y)*point.denormalization_x)
        self.sum_x = x
        self.sum_y = y

    def get_hand_lm_list(self):
        lm = []
        for point in self.hand_lm:
            lm.append(point.get_2d_coordinate())
        return lm

    def get_hand_position(self):
        return [int(self.sum_x/21), int(self.sum_y/21)]

    def get_model_parameter(self):
        return self.get_hand_lm_list()

    def get_hand_classification(self):
        return self.property_hand.index

    def get_hand_name(self):
        return self.property_hand.label


class HandProperty:
    def __init__(self, score, index, label):
        self.score = float(score)
        self.index = int(index)
        self.label = label


class Gesture:
    def __init__(self, hand, gesture_name, gesture_id):
        self.hand = hand
        self.gesture_name = gesture_name
        self.gesture_id = gesture_id


class PixelCoordinate:
    def __init__(self, normalization_x, normalization_y, normalization_z, denormalization_x,
                 denormalization_y, denormalization_z):

        self.x = int(normalization_x * denormalization_x)
        self.y = int(normalization_y * denormalization_y)
        self.z = int(normalization_z * denormalization_z)
        self.denormalization_x = denormalization_x
        self.denormalization_y = denormalization_y
        self.denormalization_z = denormalization_z

    def get_coordinate(self):
        return [self.x, self.y, self.z]

    def get_2d_coordinate(self):
        return [self.x, self.y]


class GestureFacade:
    @staticmethod
    def create_hand_model(hand_lm, property_hand):
        return HandSkeletalModel(hand_lm, property_hand)

    @staticmethod
    def create_gesture(hand_model, gesture_id, gesture_name):
        return Gesture(hand_model, gesture_name, gesture_id)

    @staticmethod
    def create_coordinate(normalization_x, normalization_y, normalization_z, denormalization_x,
                          denormalization_y, denormalization_z):
        return PixelCoordinate(normalization_x, normalization_y, normalization_z, denormalization_x,
                               denormalization_y, denormalization_z)

    @staticmethod
    def get_hand_model_parameter(hand):
        return hand.get_model_parameter()

    @staticmethod
    def get_gesture_position(gesture):
        return GestureFacade.get_hand_position(gesture.hand)

    @staticmethod
    def get_hand_position(hand):
        return hand.get_hand_position()

    @staticmethod
    def get_gesture_name(gesture):
        return gesture.gesture_name
