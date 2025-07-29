import numpy as np
import cv2

def supertrans(x,y,U):
    [Nx,Ny] = U.shape
    lx = int(np.floor(x))
    ly = int(np.floor(y))
    rx = lx + 1 if lx + 1 < Nx else lx
    ry = ly + 1 if ly + 1 < Ny else ly

    trans = U[rx,ry]*(x-lx)*(y-ly) + U[rx,ly]*(x-lx)*(ry-y) + U[lx,ry]*(rx-x)*(y-ly) + U[lx,ly]*(rx-x)*(ry-y)

    return trans

def strain_calculation(mask, V, NegV):
    BW = mask == 1
    trans_field_x = np.zeros((160, 160, 25))
    trans_field_y = np.zeros((160, 160, 25))
    trans_field_xo = np.zeros((160, 160))
    trans_field_yo = np.zeros((160, 160))
    for x in range(160):
        for y in range(160):
            x_regis = x + supertrans(x, y, NegV[0, 1, 0])
            y_regis = y + supertrans(x, y, NegV[0, 0, 0])
            trans_field_xo[x, y] = y_regis
            trans_field_yo[x, y] = x_regis
            for j in range(25):
                x_back = x_regis + supertrans(x_regis, y_regis, V[0, 1, j])
                y_back = y_regis + supertrans(x_regis, y_regis, V[0, 0, j])
                trans_field_x[x, y, j] = y_back
                trans_field_y[x, y, j] = x_back

    V2 = np.zeros((2,160,160,25))
    V2[0,:,:,:] = trans_field_x.copy()
    V2[1,:,:,:] = trans_field_y.copy()
    V1 = np.zeros((2, 160, 160))
    V1[0, :, :] = trans_field_xo.copy()
    V1[1, :, :] = trans_field_yo.copy()

    mask = cv2.convertScaleAbs(mask)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    smap1 = np.zeros((160, 160, 25))
    cir_direc_field_x = np.zeros((160, 160))
    cir_direc_field_y = np.zeros((160, 160))
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        [Y, X] = np.where(component_mask == 1)
        if len(Y) < 3:
            continue
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0][:, 0, :]

        contoury = contours[:, 1]
        contourx = contours[:, 0]
        contourxx = np.append(contourx,contourx[0])
        contouryy = np.append(contoury,contoury[0])
        contour_dirx = np.diff(contourxx)
        contour_diry = np.diff(contouryy)
        tmp = np.sqrt(contour_dirx ** 2 + contour_diry ** 2)
        contour_dirx = contour_dirx / tmp
        contour_diry = contour_diry / tmp


        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            dist = np.sqrt((x - contourx) ** 2 + (y - contoury) ** 2)
            M = round(0.4 * np.size(contourx))
            if M < 10:
                M = 10
            ind = np.argsort(dist)[:M]
            dist = dist[ind]
            temppoint = ind[0]
            cir_direc_x = 0
            cir_direc_y = 0
            for checkpoint in range(len(dist)):
                dirx = contour_dirx[ind[checkpoint]]
                diry = contour_diry[ind[checkpoint]]
                direction_vector_template = np.array([contour_dirx[temppoint], contour_diry[temppoint]])
                direction_vector = np.array([dirx, diry])
                dot_product = np.dot(direction_vector_template, direction_vector)
                if dot_product < 0:
                    cir_direc_x += -dirx
                    cir_direc_y += -diry
                else:
                    cir_direc_x += dirx
                    cir_direc_y += diry

            tmp = np.sqrt(cir_direc_x ** 2 + cir_direc_y ** 2)
            cir_direc_x = cir_direc_x / tmp
            cir_direc_y = cir_direc_y / tmp

            cir_direc_field_x[y, x] = -cir_direc_x
            cir_direc_field_y[y, x] = -cir_direc_y

    [Y, X] = np.where(mask == 1)
    M = len(X)
    direc_field1_x = cir_direc_field_x.copy()
    direc_field1_y = cir_direc_field_y.copy()

    for i in range(25):
        [F12, F11] = np.gradient(trans_field_x[:, :, i])
        [F22, F21] = np.gradient(trans_field_y[:, :, i])
        E11 = (F11 ** 2 + F21 ** 2 - 1) / 2
        E21 = (F11 * F12 + F21 * F22) / 2
        E22 = (F12 ** 2 + F22 ** 2 - 1) / 2

        for j in range(M):
            x = X[j]
            y = Y[j]
            direc1_x = direc_field1_x[y, x]
            direc1_y = direc_field1_y[y, x]

            e11 = E11[y, x]
            e21 = E21[y, x]
            e22 = E22[y, x]

            smap1[y, x, i] = e11 * (direc1_x ** 2) + 2 * e21 * direc1_y * direc1_x + e22 * (direc1_y ** 2)

    scurve1 = np.zeros((25,1))

    for i in range(25):
        tmp1 = smap1[:,:,i]
        scurve1[i] = np.mean(tmp1[BW])
    tmp1 = scurve1.copy()

    for i in range(25):
        xx = i-1 if i > 0 else 0
        yy = i+2 if i < 24 else 25
        ind = np.arange(xx,yy)
        scurve1[i] = np.mean(tmp1[ind])

    return scurve1