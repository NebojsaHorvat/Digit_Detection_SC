import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

alphabet = [0,1,2,3,4,5,6,7,8,9]
colors = cm.rainbow(np.linspace(0, 1, 10))
class Image:
    def get_color(self,j):
        j = j %10
        return colors[j]
    def load_image(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def image_gray(self,image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def image_bin(self,image_gs):
        height, width = image_gs.shape[0:2]
        image_binary = np.ndarray((height, width), dtype=np.uint8)
        #ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_OTSU)
        ret,image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_OTSU)
        return image_bin

    def invert(self,image):
        return 255-image

    def display_image(self,image, color= False):
        if color:
            plt.imshow(image)
        else:
            plt.imshow(image, 'gray')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.show()
    def dilate(self,image):
        kernel = np.ones((2,2)) # strukturni element 3x3 blok
        #kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        return cv2.dilate(image, kernel, iterations=1)

    def erode(self,image):
        kernel = np.ones((2,2)) # strukturni element 3x3 blok
        #kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        return cv2.erode(image, kernel, iterations=1)

    def resize_region(self,region):
        '''Transformisati selektovani region na sliku dimenzija 28x28'''
        return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

    def scale_to_range(self,image): # skalira elemente slike na opseg od 0 do 1
        ''' Elementi matrice image su vrednosti 0 ili 255.
            Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
        '''
        return image/255

    def select_roi(self,image_orig, image_bin):

        img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
        regions_array = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
            area = cv2.contourArea(contour)
            if area > 60 and h < 30 and h > 17 and w > 8: #if area > 60 and h < 100 and h > 15 and w > 5: ovo nije radilo u videu 3 pred kraj
                # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
                # oznaciti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
                #region = image_bin[y:y + h + 1, x:x + w + 1]
                region = image_bin[y-3:y + h + 4, x-3:x + w + 4]
                regions_array.append([self.resize_region(region), (x, y, w, h)])
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = sorted_regions = [region[0] for region in regions_array]

        # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
        return image_orig, sorted_regions, regions_array


    def matrix_to_vector(self,image):
        '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
        return image.flatten()


    def prepare_for_ann(self,regions):
        ready_for_ann = []
        for region in regions:
            # skalirati elemente regiona
            # region sa skaliranim elementima pretvoriti u vektor
            # vektor dodati u listu spremnih regiona
            scale = self.scale_to_range(region)
            ready_for_ann.append(self.matrix_to_vector(scale))

        return ready_for_ann

    def winner(self,output): # output je vektor sa izlaza neuronske mreze
        '''pronaci i vratiti indeks neurona koji je najvise pobudjen'''
        return max(enumerate(output), key=lambda x: x[1])[0]


    def display_result(self,outputs, alphabet):
        '''za svaki rezultat pronaci indeks pobednickog
            regiona koji ujedno predstavlja i indeks u alfabetu.
            Dodati karakter iz alfabet u result'''
        result = []
        for output in outputs:
            result.append(alphabet[self.winner(output)])
        return result
