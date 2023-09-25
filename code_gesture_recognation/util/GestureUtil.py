class Hand:

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
