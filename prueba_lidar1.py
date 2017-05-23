carpeta=""

import os
import glob
import re
import sys

from qgis.core import *
import qgis.utils
from qgis.utils import iface
from qgis.core import QgsProject
from PyQt4.QtCore import QFileInfo
from qgis.gui import *
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
from math import *
import processing

from PyQt4.QtCore import *
from PyQt4.QtGui import *

#hay que instalarlo en el python del qgis
#from laspy.file import File
import numpy as np
import laspy
import copy

#para sacar los histogramas
import matplotlib.pyplot as plt





#compruebo que capas estan cargadas en el proyecto al iniciar el script
capasoriginales =QgsMapLayerRegistry.instance().mapLayers()


a=["nombre de archivo","extension"]

#congelo la vista  para ahorrar memoria
canvas = iface.mapCanvas()
canvas.freeze(True)


    

source="c:/work/carpeta/recortado.las"
target="c:/work/carpeta/suelo.asc"
lassuelo="c:/work/carpeta/suelo.las"
print "bien1"

#genero un las nuevo solo con los puntos de suelo, clasificados en el archivo como tal
# Open an input file in read mode.
inFile = laspy.file.File(source,mode= "r")

# Call copy on the HeaderManager object to get a more portable Header instance.
# This means we don't  have to modify the header on the read mode inFile.
new_header = copy.copy(inFile.header)
# Update the fields we want to change, the header format and data_format_id
new_header.format = 1.1
new_header.pt_dat_format_id = 0
print "bien2"
# Now we can create a new output file with our modified header.
# Note that we need to give the file the VLRs manually, because the low level
# header doesn't know about them, while the header manager does.
    
outFile = laspy.file.File(lassuelo, mode= "w", vlrs = inFile.header.vlrs, header = new_header)
print "bien3"
# Iterate over all of the available point format specifications, attepmt to
# copy them to the new file. If we fail, print a message.

# Take note of the get_dimension and set_dimension functions. These are
# useful for automating dimension oriented tasks, because they just require
# the spec name to do the lookup.

for spec in inFile.reader.point_format:
    print("Copying dimension: " + spec.name)
    #filtro los puntos clasificados como suelo
    in_spec = inFile.reader.get_dimension(spec.name)[inFile.classification==2]
    print "bien4"
    try:
        outFile.writer.set_dimension(spec.name, in_spec)
        print "bien5"
    except(util.LaspyException):
        print("Couldn't set dimension: " + spec.name +                    " with file format " + str(outFile.header.version) +                    ", and point_format " + str(outFile.header.data_format_id))

# Close the file

outFile.close()

print "bien6"


lasnosuelo="c:/work/carpeta/nosuelo.las"

#genero un las nuevo solo con los puntos de NO suelo, clasificados en el archivo como tal
# Open an input file in read mode.
inFile = laspy.file.File(source,mode= "r")

# Call copy on the HeaderManager object to get a more portable Header instance.
# This means we don't  have to modify the header on the read mode inFile.
new_header = copy.copy(inFile.header)
# Update the fields we want to change, the header format and data_format_id
new_header.format = 1.1
new_header.pt_dat_format_id = 0
print "bien2"
# Now we can create a new output file with our modified header.
# Note that we need to give the file the VLRs manually, because the low level
# header doesn't know about them, while the header manager does.
    
outFile = laspy.file.File(lasnosuelo, mode= "w", vlrs = inFile.header.vlrs, header = new_header)
print "bien3"
# Iterate over all of the available point format specifications, attepmt to
# copy them to the new file. If we fail, print a message.

# Take note of the get_dimension and set_dimension functions. These are
# useful for automating dimension oriented tasks, because they just require
# the spec name to do the lookup.

for spec in inFile.reader.point_format:
    print("Copying dimension: " + spec.name)
    #filtro los puntos clasificados como suelo
    in_spec = inFile.reader.get_dimension(spec.name)[inFile.classification<>2]
    print "bien4"
    try:
        outFile.writer.set_dimension(spec.name, in_spec)
        print "bien5"
    except(util.LaspyException):
        print("Couldn't set dimension: " + spec.name +                    " with file format " + str(outFile.header.version) +                    ", and point_format " + str(outFile.header.data_format_id))

# Close the file

outFile.close()


#genero un suelo continuo a partir de los puntos medios de suelo de cada cuadricula de 1x1
cell=1.0
NODATA =0
#pongo como archivo de entrada el obtenido del filtrado de suelo
las=laspy.file.File(lassuelo,mode= "r")

print "bien7"


min=las.header.min
max=las.header.max
print min
print max
print "bien8"
xdist=max[0]-min[0]
ydist=max[1]-min[1]
cols=int(xdist)/cell
rows=int(ydist)/cell
cols+=1
rows+=1
count=np.zeros((rows,cols)).astype(np.float32)
zsum=np.zeros((rows,cols)).astype(np.float32)
ycell=-1*cell
projx=(las.x-min[0])/cell
projy=(las.y-min[1])/ycell
ix=projx.astype(np.int32)
iy=projy.astype(np.int32)
print "bien9"
for x,y,z in np.nditer([ix,iy,las.z]):
    count[y,x]+=1
    zsum[y,x]+=z
nonzero=np.where(count>0,count,1)
zavg=zsum/nonzero
#print zavg
#no me gusta
mean=np.ones((rows,cols))*np.mean(zavg)
left=np.roll(zavg,-1,1)
lavg=np.where(left>0,left,mean)
right=np.roll(zavg,1,1)
ravg=np.where(right>0, right,mean)
interpolate=(lavg+ravg)/2
fill=np.where(zavg>0,zavg,interpolate)
#lo intento de otra forma
from scipy.interpolate import griddata
    #grid_x, grid_y = np.mgrid[min[0]:min[0]+cols:cols, min[1]:min[1]+rows:rows]
    #grid_z0 = griddata(xy, z, (grid_x, grid_y), method='linear',fill_value=0)

header='ncols %s\n'% fill.shape[1]
header+='nrows %s\n'% fill.shape[0]
header+='xllcorner %s\n'% min[0]
header+='yllcorner %s\n'% min[1]
header+='cellsize %s\n'% cell
header+='NODATA_value %s\n'% NODATA

target="c:/work/carpeta/suelofill2.asc"
with open (target,"wb") as f:
    f.write(header)
    np.savetxt(f,fill,fmt="%1.2f")
target="c:/work/carpeta/suelomean.asc"
with open (target,"wb") as f:
    f.write(header)
    np.savetxt(f,mean,fmt="%1.2f")
target="c:/work/carpeta/suelozavg.asc"
with open (target,"wb") as f:
    f.write(header)
    np.savetxt(f,zavg,fmt="%1.2f")
target="c:/work/carpeta/nonzero.asc"
with open (target,"wb") as f:
    f.write(header)
    np.savetxt(f,nonzero,fmt="%1.2f")
target="c:/work/carpeta/zsum.asc"
with open (target,"wb") as f:
    f.write(header)
    np.savetxt(f,zsum,fmt="%1.2f")

print "bien 9.1"
#filtro para rellenar huecos pequenos
def rellenahuecos(input, output):
    #input=os.path.join(carpeta,troncoresumido+'_'+rasterdeentrada+'4.tif')
    #input="c:/work/carpeta/suelozavg.asc"
    distance=30
    iterations=0
    band=1
    mask=None
    no_default_mask='True'
    #output=os.path.join(carpeta,troncoresumido+'_'+rasterdeentrada+'5.tif')
    #output="c:/work/carpeta/fillnodata.tif"
    processing.runalg('gdalogr:fillnodata', input, distance, iterations, band,mask,no_default_mask, output)
    #StringToRaster(os.path.join(carpeta,troncoresumido+'_'+rasterdeentrada+'5.tif'),rasterdeentrada+str("5"))   
    
#primera vez de rellenar huecos
input="c:/work/carpeta/suelozavg.asc"
output="c:/work/carpeta/fillnodata.tif"
rellenahuecos(input, output)
print "bien 9.2"

#cargo el tif con el suelo que acabo de generar

 

 
#Raster
fileName = output
fileInfo = QFileInfo(fileName)
baseName = fileInfo.baseName()
rlayer = QgsRasterLayer(fileName, baseName)
if not rlayer.isValid():
  print "Layer failed to load"
 
#Add layer

QgsMapLayerRegistry.instance().addMapLayer(rlayer)




#leo los valores del tif del suelo
from osgeo import gdal
import os
import struct
 
#layer = iface.activeLayer()
layer=rlayer
provider = layer.dataProvider()
 
fmttypes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 'Int32':'i', 'Float32':'f', 'Float64':'d'}
 
path= provider.dataSourceUri()
 
(raiz, filename) = os.path.split(path)
 
dataset = gdal.Open(path)
 
band = dataset.GetRasterBand(1)
   
totHeight = 0
 
print "filas = %d columnas = %d" % (band.YSize, band.XSize)
 
BandType = gdal.GetDataTypeName(band.DataType)
   
print "Tipo de datos = ", BandType
  
print "Ejecutando estadisticas de %s" % filename
print "que se encuentra en %s" % raiz

for y in range(band.YSize):
   
    scanline = band.ReadRaster(0, y, band.XSize, 1, band.XSize, 1, band.DataType)
    values = struct.unpack(fmttypes[BandType] * band.XSize, scanline)
   
    for value in values:
        totHeight += value
           
average = totHeight / float((band.XSize * band.YSize))
   
print "Promedio = %0.5f" % average
  
dataset = None



    
#cargo el tif en un vector porque de la manera anterior no puedo sacar el xyx
ds = gdal.Open(output)
myarray = np.array(ds.GetRasterBand(1).ReadAsArray())


    
    
    
    #normalizo los puntos de no suelo, restandoles el suelo de la cuadricula que acabo de generar.
#abro el lasnosuelo
lasnosuelofile=laspy.file.File(lasnosuelo,mode= "r")
print "bien 10"
#recorro todos sus elementos
    
 #creo una capa de puntos temporal con los resultados
# create layer
#from PyQt4.QtCore import QVariant
fields=QgsFields()
fields.append(QgsField("min", QVariant.String))
fields.append(QgsField("per10", QVariant.String))
fields.append(QgsField("per20", QVariant.String))
fields.append(QgsField("per30", QVariant.String))
fields.append(QgsField("per10090", QVariant.String))
fields.append(QgsField("per50", QVariant.String))
fields.append(QgsField("ultimet", QVariant.String))
fields.append(QgsField("ult2met", QVariant.String))
fields.append(QgsField("relacion", QVariant.String))
fields.append(QgsField("per90", QVariant.String))
fields.append(QgsField("per100", QVariant.String))
fields.append(QgsField("enlace", QVariant.String))

writer=QgsVectorFileWriter("c:/work/carpeta/puntos.shp","CP1250",fields,QGis.WKBPoint,None,"ESRI Shapefile")
#vl = QgsVectorLayer("Point", "temporary_points", "memory")
#pr = vl.dataProvider()
print "ok creada la capa y los campos"

#funcion para hacer regresion polinomica y  que nos de el r2
# Polynomial Regression
def polyfit2(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

#funcion que genera una capa de puntos con datos de percentiles
def generapunto(min,per10,per20,per30,per10090,per50,ultimet,r2,relacion,per90,per100,nombrehistograma,x,y):

    #vl.startEditing()
    # add fields
    #pr.addAttributes([,QgsField("per20", QVariant.Double),QgsField("per30", QVariant.Double),QgsField("per40", QVariant.Double),QgsField("per50", QVariant.Double),QgsField("per60", QVariant.Double),QgsField("per70", QVariant.Double),QgsField("per80", QVariant.Double),QgsField("per90", QVariant.Double),QgsField("per100", QVariant.Double),                    QgsField("x",  QVariant.Int),                    QgsField("y", QVariant.Double)])
    #vl.updateFields() 
    # tell the vector layer to fetch changes from the provider
    #print "ok creados los campos"
    # add a feature
    #print min,per10,per20,per30,per40,per50,per60,per70,per80,per90,per100,x,y
    fet = QgsFeature()
    fet.setGeometry(QgsGeometry.fromPoint(QgsPoint(x,y)))
    fet.setAttributes([min,per10,per20,per30,per10090,per50,ultimet,r2,relacion,per90,per100,nombrehistograma,x,y])
    writer.addFeature(fet)
    
 
    #cambio la simbologia
    symbol = QgsMarkerSymbolV2.createSimple({'name': 'circle', 'color': 'orange','size': '5',})
    #vl.rendererV2().setSymbol(symbol)
    #la convierto en shape
    #cLayer = qgis.utils.iface.mapCanvas().currentLayer()
    #provider = cLayer.dataProvider()
    #writer = QgsVectorFileWriter.writeAsVectorFormat( "c:/work/carpeta/puntos.shp", pr.encoding(), pr.fields(),"ESRI Shapefile" )
    
    #QgsMapLayerRegistry.instance().addMapLayer(vl)  

    # update layer's extent when new features have been added
    # because change of extent in provider is not propagated to the layer
    #vl.updateExtents()
    #vl.commitChanges()
    #vl.updateExtents()
    #canvas = qgis.utils.iface.mapCanvas()
    #canvas.setExtent(vl.extent())
    #vl.updateFieldMap()   
    
try:
    normalizado=np.array([[0,0,0]])
    for x,y,z in np.nditer([lasnosuelofile.x, lasnosuelofile.y, lasnosuelofile.z]):
        ident=rlayer.dataProvider().identify(QgsPoint(x, y), QgsRaster.IdentifyFormatValue)
        zsuelo= ident.results()[1]

    #zsuelo=myarray[int(x-x0),int(y0-y)]
    #print zsuelo
        if type(zsuelo) not in (int, float):
            zsuelo=z
        znormalizada= z-zsuelo
    #matriz con las x y z de los puntos no clasificados como suelo siendo la z la altura respecto al suelo
        normalizado=np.append(normalizado, [[x, y, znormalizada]], axis=0)
    #print normalizado  
#genero la cuadricula de estudio que tiene que ir variando de 10x10m
    a=[]
#xi=523447
#yi=4660530
#creo una celda de 10x10 y va cambiando
    for xi in range(int(min[0]),int(max[0]),10):
        for yi in range(int(min[1]),int(max[1]),10):
            for x,y,z in np.nditer([normalizado[:,0], normalizado[:,1], normalizado[:,2]]):
                if xi<=x<xi+10 and yi<=y<yi+10:
                    a.append(float(z))
            print "bien 10 bis"


        

            b=np.array(a)
            #print b imprime todo el listado de puntos de la celda
            print "listado completo de puntos"
            print b
            
            #saco el numero de puntos del ultimo medio metro
            ultimomediometro=np.amax(b)-0.35
            c=[]
            for punto in b:
                if punto>=ultimomediometro:
                    c.append(punto)
            nptosultimomediometro= len(c)
            #saco el numero de puntos de los ultimo metro
            ultimometro=np.amax(b)-0.70
            d=[]
            for punto in b:
                if ultimomediometro>punto>=ultimometro:
                    d.append(punto)
            nptosultimometro= len(d)
            #saco el numero de puntos de los ultimo metro
            ultimometroymedio=np.amax(b)-1.05
            e=[]
            for punto in b:
                if ultimometro>punto>=ultimometroymedio:
                    e.append(punto)
            nptosultimometroymedio= len(e)
            #saco el numero de puntos de los ultimos 2 metros
            ultimos2metros=np.amax(b)-1.40
        
            f=[]
            for punto in b:
                if ultimometroymedio>punto>=ultimos2metros:
                    f.append(punto)
            nptosultimos2metros= len(f)
        
            ultimos2metrosypico=np.amax(b)-1.75
            h=[]
            for punto in b:
                if ultimos2metros>punto>=ultimos2metrosypico:
                    h.append(punto)
            nptosultimos2metrosypico= len(h)
        
            numerodepuntos=[nptosultimomediometro,nptosultimometro,nptosultimometroymedio,nptosultimos2metros,nptosultimos2metrosypico]
            rango=[ultimomediometro,ultimometro,ultimometroymedio,ultimos2metros,ultimos2metrosypico]
            alturaaa=np.amax(b)-1.50
            g=[]
            for punto in b:
                if punto>=alturaaa:
                    g.append(punto)
                
                
            lenalturaaa= len(g)
            yyy=numerodepuntos
            xxx=rango
 
            resultado=np.polyfit(xxx,yyy,2)
            resultado2=polyfit2(xxx,yyy,2)
            f_resultado=np.poly1d(resultado)
            print xi, yi
            print f_resultado
            print b
        #print listado
            print "resultado"

            print resultado
        #print str(nptosultimomediometro)+"_"+str(  nptosultimometro )+"_"+str( nptosultimometroymedio  )+"_"+str(nptosultimos2metros)           
            
        




        





        
        

        
            relacion=0.00

            relacion =str(resultado2['polynomial'][0])
            print relacion
            #genero un histograma
            r2=str(resultado2['determination'])
            print "r2"+str(r2)[0]
            #listado=[np.amin(b),np.percentile(b,10),np.percentile(b,20),np.percentile(b,30),np.percentile(b,40),np.percentile(b,50),np.percentile(b,60),np.percentile(b,70),np.percentile(b,80),np.percentile(b,90),np.percentile(b,100),np.amax(b)]
        
            listado=[np.amin(b),np.percentile(b,10),np.percentile(b,20),np.percentile(b,30),np.percentile(b,90)-np.percentile(b,100),np.percentile(b,50),nptosultimometro,r2,relacion,np.percentile(b,90),np.percentile(b,100),np.amax(b)]
            print "bien 10 ter"
        
            
            plt.hist(b, bins=[0,0.5,1.0,1.5,2.0,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27])
        
            plt.title("x"+str(xi)+"y"+str(yi))
            print "bien 10 quart"
        #plt.show()
            nombrehistograma="c:/work/carpeta/histograma"+str(xi)+str(yi)+".png"
            plt.savefig(nombrehistograma) 
            plt.clf()
            plt.close()
            
            #genero una imagen con las z

            plt.plot(xxx, yyy, 'ro')
 #          xx=np.arange(0,27,1)
            print "bien 10 cinc"
            nuevasx=np.array([np.amax(b),np.amax(b)-0.35,np.amax(b)-0.7,np.amax(b)-1.05,np.amax(b)-1.4,np.amax(b)-1.75])
            print nuevasx
            plt.plot(nuevasx,f_resultado(nuevasx))
            print "bien 10 six"    
            #plt.axis([0, 6, 0, 20])
            #plt.show()
            plt.title("x"+str(xi)+"y"+str(yi)+" "+str(relacion)+"_"+str(r2))
            #plt.show()
            nombrearchivo="c:/work/carpeta/picota"+str(xi)+str(yi)+".png"
            plt.savefig(nombrearchivo) 
            plt.clf()
            plt.close()
            print "bien 10 sev"    
            a=[]
                #generapunto(np.amin(b),np.percentile(b,10),np.percentile(b,20),np.percentile(b,30),np.percentile(b,40),np.percentile(b,50),np.percentile(b,60),np.percentile(b,70),np.percentile(b,80),np.percentile(b,90),np.percentile(b,100),x,y)    
            #print listado[0],listado[1],listado[2],listado[3],listado[4],listado[5],listado[6],listado[7],listado[8],listado[9],listado[10],xi+5,yi+5
        
            generapunto(str(listado[0]),str(listado[1]),str(listado[2]),str(listado[3]),str(listado[4]),str(listado[5]),str(listado[6]),str(listado[7]),str(listado[8]),str(listado[9]),str(listado[10]),nombrehistograma,xi+5,yi+5)
except:
    pass





    
print "biene 11"

    
    
    



del writer
    
    
    

    
    
    
    
    
    
"""    
   
#elimino las capas que he cargado durante el proceso
capas =QgsMapLayerRegistry.instance().mapLayers()
for capa in capas:
    if capa not in capasoriginales:
        QgsMapLayerRegistry.instance().removeMapLayers( [capa] )
del(capas)"""
        

#repinto todo refrescando la vista
canvas.freeze(False)
canvas.refresh()
