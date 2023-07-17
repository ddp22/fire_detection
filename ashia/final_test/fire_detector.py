import torch
from fire_classifier import FireClassifier
import albumentations
import cv2

class FireDetector:
    '''Classe che astrae la rete neurale. Contiene un costruttore e 
    un metodo per l'analisi di un singolo frame'''

    def __init__(self, net_model, weights_path):
       '''Il costruttore prende due attributi: 
       - net_model è l'oggetto che contiene il modello della rete
       - weights_path sono i pesi da caricare sul modello'''

       self.model = net_model.cuda()  # Importa il modello
       self.model.load_state_dict(torch.load((weights_path)))  # Caricamento dei pesi migliori
       self.model.eval()  # Modello settato per l'utilizzo
       self.list = []

    def empty_list(self):
        self.list = []

    def detect_fire(self, img):
        # PREPARAZIONE DELL'IMMAGINE
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessing = albumentations.Compose([
            albumentations.Resize(height=224, width=224, always_apply=True), # resize
            albumentations.Normalize(mean=[0.485, 0.456, 0.406],   # normalizzazione
                                    std=[0.229, 0.224, 0.225],
                                    max_pixel_value=255.,
                                    always_apply=True),
            ])
        preprocessed = preprocessing(image=image)
        preprocessed_image = preprocessed["image"]
        tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).cuda()
        # CLASSIFICAZIONE DELL'IMMAGINE
        with torch.no_grad():
            output = self.model(tensor)
        o = torch.where(output > .5, 1, 0).int()
        return o
    
    def final_results(self, fps=1, consecutive_frames=4):
        '''Questa funzione prende in ingresso una lista di 0/1 e 
        implementa una policy per decidere il secondo in cui 
        si individua il fuoco.
        Restituisce 0 o 1 per indicare la presenza del fuoco e 
        un secondo intero che rappresenta il secondo in cui il fuoco 
        viene individuato'''
        i = 0  # numero di 1 consegutivi trovati 
        frame = 0
        for out in self.list:
            if out == 1:  # se è stato individuato il fuoco 
                i = i + 1     # incrementa il contatore 
            else:  # Altrimenti 
                i = 0  # lo azzera di nuovo 

            frame = frame+1  # incrementa l'indice del frame
            if i >= consecutive_frames:
                self.empty_list()  # Svuota la lista di frames
                return 1, (frame/fps).int()
        self.empty_list()
        return 0, 1


       
