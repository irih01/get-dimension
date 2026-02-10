import cv2
import os
from PIL import Image


class GenerateAruco:
    def __init__(self, dictionary_type = cv2.aruco.DICT_4X4_50):
        """
        Iniţializează clasa cu un anumit tip de dicţionar ArUco.
        4x4 50 default
        """
        self.aruco = cv2.aruco
        self.dictionary = self.aruco.getPredefinedDictionary(dictionary_type)


    
    def generate_marker(self, marker_id: int = 0, size: int = 200, output_path: str = "marker.png"):
        """
        Genereaza un marker ArUco si il salveaza intr-un fisier imagine.

        :param marker_id: ID_ul markerului (0 - nr maxim din dictionar -1)
        :param size: Dimenisunea imaginii (px)
        :param output_path: Numele fisierului
        """
        
        marker_image = self.aruco.generateImageMarker(self.dictionary, marker_id, size)

        # Creare folder
        #os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        cv2.imwrite(output_path, marker_image)
        print (f'[INFO] Marker salvat la: {output_path}')

        return marker_image
    


    def multi_markers(self, start_id: int = 0, end_id: int = 9, size: int = 200, output_folder: str = 'markers'):
        """ 
        Genereaza mai multi markeri ArUco si ii salveaza intr-un folder.
        :param start_id: Primul ID de marker (implicit 0)
        :param end_id: Ultimul ID de marker (implicit 9)
        :param output_folder: Numele folderului (markers)
        """

        os.makedirs(output_folder, exist_ok=True)

        for marker_id in range(start_id, end_id + 1):
            filename = os.path.join(output_folder, f'marker{marker_id}.png')
            self.generate_marker(marker_id, size, filename)
        print(f'[INFO] Markere generate: {start_id} -> {end_id} (salvate in {output_folder})')





    def generate_pdf(self, output_folder: str = 'markers', pdf_path: str = 'aruco_markers.pdf',
                     markers_per_row: int = 3, marker_size: int = 200, padding: int = 20):
        """Creaza un PDF cu toti markerii dintr-un folder, aranjati in grila"""  
        marker_files = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.png')])
        if not marker_files:
            print("[WARN] Niciun marker gasit in folder!")
            return
        
        images = [Image.open(f).resize((marker_size, marker_size)) for f in marker_files]
        num_markers = len(images)
        rows = (num_markers + markers_per_row - 1) // markers_per_row
        cols = markers_per_row

        width = cols * (marker_size + padding) + padding
        height = rows * (marker_size + padding) + padding

        pdf_image = Image.new("RGB", (width, height), color='white')

        x, y = padding, padding
        for idx, img in enumerate(images):
            pdf_image.paste(img, (x,y))
            x += marker_size + padding

            if (idx + 1) % cols == 0:
                x = padding
                y += marker_size + padding

        pdf_image.save(pdf_path, 'PDF')
        print(f'[INFO] PDF generat: {pdf_path}')





    
    def show_marker(self, marker_image, window_name: str = 'ArUco Marker'):
        """
        Afiseaza markerul intr-o fereastra.
        """
        cv2.imshow(window_name, marker_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    generator = GenerateAruco(cv2.aruco.DICT_4X4_50)
    marker = generator.generate_marker(0, 200, 'marker0.png')
    generator.show_marker(marker)