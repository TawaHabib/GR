class GestureUtil:

    def __init__(self, posizione, gesture_name, h, larghezza):
        self.posizione = posizione
        self.puntoMano = [20, 20]
        self.x = 0
        self.y = 0
        self.h = h
        self.larghezza = larghezza
        for e in self.posizione:
            for j in e:
                self.x = self.x + int((j[0]/h)*larghezza)
                self.y = self.y + int((j[1]/larghezza)*h)
        self.x = int(self.x/21)
        self.y = int(self.y/21)
        self.puntoMano = [self.x, self.y]
        self.gestureName = gesture_name
