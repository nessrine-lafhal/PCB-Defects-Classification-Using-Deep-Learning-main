import cv2
from PIL import Image, ImageOps
import numpy as np
import imutils
import matplotlib.pyplot as plt
from keras.models import load_model

def morph_transform(ref, test): 
    # La fonction morph_transform applique une transformation morphologique pour aligner deux images

    img1 = test  # Image de test
    img2 = ref  # Image de référence
    height, width = img2.shape[:2]  # Récupération des dimensions de l'image de référence

    orb_detector = cv2.ORB_create(5000)  # Création d'un détecteur ORB avec 5000 points de caractéristiques
    kp1, d1 = orb_detector.detectAndCompute(img1, None)  # Détection et calcul des descripteurs pour l'image de test
    kp2, d2 = orb_detector.detectAndCompute(img2, None)  # Détection et calcul des descripteurs pour l'image de référence

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Création d'un matcher avec la norme Hamming
    matches = matcher.match(d1, d2)  # Correspondances entre les descripteurs des deux images

    matches = sorted(matches, key=lambda x: x.distance)  # Tri des correspondances par distance
    matches = matches[:int(len(matches) * 90)]  # On garde 90% des meilleures correspondances

    no_of_matches = len(matches)  # Nombre de correspondances retenues

    p1 = np.zeros((no_of_matches, 2))  # Tableau pour les points de l'image de test
    p2 = np.zeros((no_of_matches, 2))  # Tableau pour les points de l'image de référence

    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt  # Récupération des coordonnées des points de l'image de test
        p2[i, :] = kp2[matches[i].trainIdx].pt  # Récupération des coordonnées des points de l'image de référence

    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)  # Calcul de l'homographie entre les deux jeux de points
    transformed_img = cv2.warpPerspective(test, homography, (width, height))  # Application de l'homographie pour transformer l'image de test
    return transformed_img  # Retourne l'image transformée

# Chargement des images en niveaux de gris
image1 = cv2.imread(r'C:\Users\Nessrine\Desktop\my S4\Gestion industrielle\projet industriel\PCB-Defects-Classification-Using-Deep-Learning-main\PCB-Defects-Classification-Using-Deep-Learning-main\Code\TemplatePath\1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r'C:\Users\Nessrine\Desktop\my S4\Gestion industrielle\projet industriel\PCB-Defects-Classification-Using-Deep-Learning-main\PCB-Defects-Classification-Using-Deep-Learning-main\Code\TestPath\2.png', cv2.IMREAD_GRAYSCALE)

# Application de la transformation morphologique pour aligner les images
ref_test = morph_transform(image1, image2)
image2 = ref_test

# Application d'un filtre médian pour réduire le bruit
image1 = cv2.medianBlur(image1, 5)
image2 = cv2.medianBlur(image2, 5)

# Calcul de l'opération XOR bit à bit pour mettre en évidence les différences entre les images
image_res = cv2.bitwise_xor(image1, image2)

# Affichage du résultat de l'opération XOR
cv2.imshow('RES_XOR', image_res)
cv2.waitKey(0)

# Filtrage de l'image résultante avec un filtre médian
image_res = cv2.medianBlur(image_res, 5)

# Création d'éléments structurants pour les transformations morphologiques
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Application de la fermeture morphologique pour combler les trous
image_res = cv2.morphologyEx(image_res, cv2.MORPH_CLOSE, kernel1)

# Application de l'ouverture morphologique pour enlever le bruit
image_res = cv2.morphologyEx(image_res, cv2.MORPH_OPEN, kernel2)

# Seuillage de l'image pour obtenir une image binaire
thresh, image_res = cv2.threshold(image_res, 125, 255, cv2.THRESH_BINARY)

# Affichage de l'image binaire après filtrage
cv2.imshow('RES_XOR_AFTERFILT', image_res)
cv2.waitKey(0)

# Détection des contours avec l'algorithme de Canny
edges = cv2.Canny(image_res, 30, 200)

# Affichage des contours détectés
cv2.imshow('RES_CONTOURS', edges)
cv2.waitKey(0)

# Détection des contours dans l'image binaire
cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Rechargement de l'image de test originale
img2 = cv2.imread(r'C:\Users\Nessrine\Desktop\my S4\Gestion industrielle\projet industriel\PCB-Defects-Classification-Using-Deep-Learning-main\PCB-Defects-Classification-Using-Deep-Learning-main\Code\TestPath\2.png', cv2.IMREAD_GRAYSCALE)

# Initialisation des listes pour les coordonnées et les centres des contours
X = []
Y = []
CX = []
CY = []
C = []


# Pour chaque contour détecté
for c in cnts:
    # Calcul des moments pour chaque contour
    M = cv2.moments(c)
    
    # Si le moment spatial n'est pas nul
    if M["m00"] != 0:
        # Calcul des coordonnées du centre de masse du contour
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Ajout des coordonnées du centre de masse aux listes CX et CY
        CX.append(cx)
        CY.append(cy)
        
        # Ajout des coordonnées du centre comme tuple à la liste C
        C.append((cx, cy))

# Affichage des listes CX et CY contenant les coordonnées des centres des contours
print(CX)
print(CY)

# Affichage de l'image de test originale
implot = plt.imshow(img2)

# Superposition des centres des contours sur l'image avec des points rouges
plt.scatter(CX, CY, c='r', s=40)
plt.show()

# Ouverture de l'image de test avec PIL et conversion en niveaux de gris
im = Image.open(r"C:\Users\Nessrine\Desktop\my S4\Gestion industrielle\projet industriel\PCB-Defects-Classification-Using-Deep-Learning-main\PCB-Defects-Classification-Using-Deep-Learning-main\Code\TestPath\2.png").convert('L') 

# Chargement du modèle de classification sauvegardé
model = load_model(r'C:\Users\Nessrine\Desktop\my S4\Gestion industrielle\projet industriel\PCB-Defects-Classification-Using-Deep-Learning-main\PCB-Defects-Classification-Using-Deep-Learning-main\SavePath.keras')

# Affichage du résumé du modèle
print(model.summary())

# Dictionnaire des classes de défauts avec leurs descriptions
classes = {
    0: "Open",
    1: "Short",
    2: "Mousebite",
    3: "Spur",
    4: "Copper",
    5: "Pin-Hole"
}

# Initialisation des listes pour les prédictions et les confiances
pred = []
confidence = []

# Pour chaque centre de contour détecté
for c in C:
    # Extraction d'une région de 64x64 pixels autour du centre du contour
    im1 = im.crop((c[0] - 32, c[1] - 32, c[0] + 32, c[1] + 32))
    im1 = np.array(im1)
    
    # Si l'image est en 2D, ajout d'une dimension pour correspondre aux dimensions d'entrée du modèle
    if len(im1.shape) == 2:
        im1 = np.expand_dims(im1, axis=2)
    
    # Ajout d'une dimension pour correspondre au batch size
    im1 = np.expand_dims(im1, axis=0)
    
    # Affichage des dimensions de l'image d'entrée
    print(im1.shape)
    
    # Prédiction de la classe de défaut pour l'image d'entrée
    a = model.predict(im1, verbose=1, batch_size=1)
    
    # Ajout de la classe prédite à la liste pred
    pred.append(np.argmax(a))
    
    # Ajout de la confiance associée à la prédiction à la liste confidence
    confidence.append(a)

# Affichage de l'image de test originale
plot_final = plt.imshow(img2)

# Superposition des centres des contours sur l'image avec des points rouges
plt.scatter(CX, CY, c='r', s=4)

# Annotation des centres des contours avec les classes de défauts et les confiances
for i, txt in enumerate(pred):
    plt.annotate([classes[txt], confidence[i][0][txt]], (CX[i], CY[i]), color='r')

# Affichage final de l'image annotée
plt.show()
