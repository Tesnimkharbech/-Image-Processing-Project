import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#*****************************************************PARTIE I*********************************************************
path = r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\MiniProjet\liftingbodybruite.png'
img = cv.imread(path,0)
#Fonction utilisée pour inverser une image binaire (masque)
def inverse(im):
    for y in range(im.shape[1]):
        for x in range(im.shape[0]):
            if (im[x, y]==255):
                im[x, y]=0
            else:
                im[x, y] = 255
    return im
#J'ai choisie de débruiter l'image dans le domaine fréquentiel et d'appliquer un filtre coupe-bande
#afin de supprimer une composante harmonique pure
#1. On applique d'abord la transformée de Fourier Discrète:
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
rows, cols = img.shape
#2. Créer un masque:
# f1=plt.figure(1)
# plt.title('histogramme')
# H = plt.hist(magnitude_spectrum.ravel(),256,[0,256]);
# plt.show()
#D'aprés l'histogramme j'ai choisi le seuil de binarisation du spectre d'amplitude = 225
#On applique un filtre moyenneur (passe bas) sur le spectre d'amplitude pour réduire le bruit, dégrader et lisser les contours
magnitude_spectrum= cv.blur(magnitude_spectrum,(2,2))
#On binarise le spectre
ret, T = cv.threshold(magnitude_spectrum, 225, 255, cv.THRESH_BINARY)

#On utilise des transformations morphologiques pour obtenir le masque
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))
kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))

ouverture = cv.morphologyEx(T, cv.MORPH_OPEN, kernel)
dilatation = cv.dilate(ouverture, kernel, iterations=1)
ouverture = cv.morphologyEx(dilatation, cv.MORPH_OPEN, kernel1)

i2 = cv.morphologyEx(ouverture, cv.MORPH_OPEN, kernel2)
i=inverse(ouverture)
i = cv.bitwise_or(i, i2)

i = cv.morphologyEx(i, cv.MORPH_CLOSE, kernel1)
i = cv.erode(i, kernel2, iterations=1)

cv.imshow('masque obtenu', i)
enrg = cv.imwrite(r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\masquePartieI.jpg',i)
if enrg:
 print('Image du masque est enregistrée avec succés.')
cv.waitKey(0)

mask = np.zeros((rows,cols,2),np.uint8)
mask[:, :,0] = i[:,:]
mask[:, :,1] = i[:,:]
#3. On applique le masque et la transformée de fourier inverse:
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
#On applique un filtre médian pour éliminer le bruit impulsionnel
img_back=cv.medianBlur(img_back,5)
#Résultat:
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Image en entrée'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Image filtrée'), plt.xticks([]), plt.yticks([])
plt.show()
#*****************************************************PARTIE II*********************************************************
path2 = r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\MiniProjet\cartebruitee.png'
crt = cv.imread(path2,0)
#On applique un filtre médian pour éliminer le bruit
crtmed=cv.medianBlur(crt,3)

cv.imshow('Carte apres un filtrage median',crtmed)
#Sauvegarder l’image filtrée dans un fichier
enrg = cv.imwrite(r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\cartefiltreePartieII.jpg',crtmed)
if enrg:
 print('Image de la carte filtrée est enregistrée avec succés.')
cv.waitKey(0);
#On affiche les dimensions de la carte pour identifier celle de la région d'interet
dimensions = crtmed.shape
print('Dimensions : ', dimensions)

roi=crtmed[220:400, 200:999]
cv.imshow('Region d\'interet',roi)
enrg = cv.imwrite(r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\ROIcrtmedPartieII.jpg',roi)
if enrg:
 print('Image de la région d\'interet est enregistrée avec succés.')
cv.waitKey(0);
#On applique une transformation Chapeau Haut de Forme pour obtenir le texte
kernel3 = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
CHF = cv.morphologyEx(roi, cv.MORPH_TOPHAT, kernel3)
cv.imshow('CHF', CHF)
cv.waitKey(0);
#On cherche un seuil de binarisation du texte
f1=plt.figure(1)
plt.title('Histogramme')
H = plt.hist(CHF.ravel(),256,[0,256]);
plt.show()
#On binarise la région d'interet avec un seuil = 60
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
ret, T = cv.threshold(CHF, 60, 255, cv.THRESH_BINARY)
#Et on applique une ouverture pour éliminer les petits détails (en dessous de "SPEECH")
I = cv.morphologyEx(T, cv.MORPH_OPEN, kernel)
cv.imshow('Image binaire', I)
enrg = cv.imwrite(r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\ROIbinPartieII.jpg',I)
if enrg:
 print('Image de la région d\'interet binarisée est enregistrée avec succés.')
#Détection des contours avec un filtre passe haut
kernel4 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
Ic = cv.filter2D(src=I, ddepth=-1, kernel=kernel4)
cv.imshow('Detection des contours avec un filtre passe haut',Ic)
enrg = cv.imwrite(r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\ROIcontoursartieII.jpg',Ic)
if enrg:
 print('Image du contour de la région d\'interet est enregistrée avec succés.')
cv.waitKey(0)
#*****************************************************PARTIE III*********************************************************
path3 = r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\MiniProjet\DrapeauAllemagne.png'
img = cv.imread(path3)
#Pour connaitre les codes BGR des couleurs du drapeau d'Allemagne :
# print(img)
#On remplace chaque couleur par son équivalent du drapeau de la Lituanie :
img[np.where((img==[0,0,0]).all(axis=2))]=[3,240,253]     #Noir  -> Jaune
img[np.where((img==[0,0,221]).all(axis=2))]=[80,240,40]   #Rouge -> Vert
img[np.where((img==[0,206,255]).all(axis=2))]=[35,29,234] #Jaune -> Rouge
#Pour éliminer le countour entre 2 couleurs successives, on applique une ouverture puis une fermeture:
kernel = cv.getStructuringElement(cv.MORPH_RECT, (50,50))
#Supprimer le countour jaune
img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
#Supprimer le countour noir
img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

cv.imshow('Drapeau de la Lituanie',img)

# enrg = cv.imwrite(r'C:\Users\Lenovo\Documents\iia4\Sem1\trait dimg\DrapLitpartieIII.jpg',img)
# if enrg:
#  print('Image du drapeau de la Lituanie est enregistrée avec succés.')

cv.waitKey(0)
# closing all open windows
cv.destroyAllWindows()
