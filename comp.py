def shapeDetermine(degree, contour):
    uerr = 1.1
    derr = 0.90
    print(degree)
    if  (degree > 60*derr and degree < 60*uerr):
        cv2.drawContours(img, [contour], 0, (255, 255, 0), -1)
        
    elif (degree > 90*derr and degree < 90*uerr):
        cv2.drawContours(img, [contour], 0, (0, 0, 200), -1)
    
    elif (degree > 108*derr and degree < 108*uerr):
        cv2.drawContours(img, [contour], 0, (255,105,180), -1)
        
    elif (degree > 120*derr and degree < 120*uerr):
        cv2.drawContours(img, [contour], 0,(255,69,0), -1)
    else:
        cv2.drawContours(img, [contour], 0,(0,200,0), -1)


def histogram_(angs):
    #Gerekli değişkenler tanımlandı
    angles = np.round((np.array(angs)))
    hist_list = []
    bins=np.linspace(0,255,256,dtype='uint8')
        
    #Her piksel değerinden kaç adet olduğu sayıldı. Histgoram hesabı    
    for k in range(len(bins)):
        mask = (angles==bins[k])
        hist_list.append(len(angles[mask]))
        
    result =np.where(hist_list == np.amax(hist_list))
    print("tek aciééé " , max(result[0]))
    return max(result[0])
