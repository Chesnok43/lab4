import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def gray(img):
	gr = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gr





def Gabor(size=111, Sigma=1.5, Gamma=1.2, Liambda=3, Pce=0, ug=0):
	
	dim = size // 2

	
	gabor = np.zeros((size, size), dtype=np.float32)

	
	for y in range(size):
		for x in range(size):
			
			px = x - dim
			py = y - dim

			
			te = ug / 180. * np.pi

			
			xx = np.cos(te) * px + np.sin(te) * py

			
			yy = -np.sin(te) * px + np.cos(te) * py

			
			gabor[y, x] = np.exp(-(xx**2 + Gamma**2 * yy**2) / (2 * Sigma**2)) * np.cos(2*np.pi*xx/Liambda + Pce)

	
	gabor /= np.sum(np.abs(gabor))

	return gabor





def GaborFilter(gr, size=111, Sigma=1.5, Gamma=1.2, Liambda=3, Pce=0, ug=0):
   
    H, W = gr.shape



    
    res = np.zeros((H, W), dtype=np.float32)

    
    gabor = Gabor(size=size, Sigma=Sigma, Gamma=Gamma, Liambda=Liambda, Pce=0, ug=ug)
    plt.imshow(gabor)
    plt.show()
        
    
    
    res = cv.filter2D(gr, -1, gabor)

    res = np.clip(res, 0, 255)
    res = res.astype(np.uint8)

    return res





def process(img):
   
    H, W, _ = img.shape

    
    gr = gray(img).astype(np.float32)

    
    angles = [0,30,60,90,120,150]

   

    res = np.zeros([H, W], dtype=np.float32)

    
    for i, A in enumerate(angles):
        
        resu = GaborFilter(gr, size=111, Sigma=1.5, Gamma=1.2, Liambda=3, ug = A)
        
        plt.imshow( resu, cmap = 'gray')
        plt.show()

        
        res += resu

    
    res = res / res.max() * 255
    res = res.astype(np.uint8)

    return res





for i in range(0,4):  
    img = cv.imread(str(i)+'.jpg').astype(np.float32)
    res = process(img)
    cv.imwrite(str(i)+"result.jpg", res)
    