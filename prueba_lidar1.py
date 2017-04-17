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






#compruebo que capas estan cargadas en el proyecto al iniciar el script
capasoriginales =QgsMapLayerRegistry.instance().mapLayers()


a=["nombre de archivo","extension"]

#congelo la vista  para ahorrar memoria
canvas = iface.mapCanvas()
canvas.freeze(True)

#defino la funcion que busca los archivos las o laz que existan y le paso los parametros resultantes del formulario
def buscalidaryejecuta(carpeta):
    for base, dirs, files in os.walk(carpeta):
        carpetas_y_subcarpetas=base
        archivos=files
        for archivo in archivos:
            a=list(os.path.splitext(archivo))
            extension=a[1].lower()
            print extension
            if extension==".laz" or extension==".las":
                b=os.path.join(base,a[0]+a[1])
                las = os.path.join(a[0]+a[1])
                #ejecuto el exprimelidar
                exprimelidar(las, carpeta)
                
#defino la funcion que lo hace todo con un archivo las o laz concreto
def exprimelidar(las, carpeta):
    fcstring = ""
    
    #defino un par de variables con el nombre del archivo y su abreviatura. Pensado para la denominacion estandar de los archivos LiDAR del PNOA
    tronco=las[:-4]
    troncoresumido=las[24:27]+"_"+las[28:32]
    

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
print "Ysize " , range(band.YSize)
print "Xsize " , range(band.XSize)
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
xi=523447
yi=4660530
for x,y,z in np.nditer([normalizado[:,0], normalizado[:,1], normalizado[:,2]]):
    #creo una celda de 10x10 y va cambiando


    if xi<=x<xi+10 and yi<=y<yi+10:
        a.append(float(z))
print "bien 10 bis"




b=np.array(a)
print b

print np.amax(b)
print np.amin(b)
print "percentil 10" 
print np.percentile(b,10)
print "percentil 20" 
print np.percentile(b,20)
print "percentil 30" 
print np.percentile(b,30)
print "percentil 40" 
print np.percentile(b,40)
print "percentil 50" 
print np.percentile(b,50)
print "percentil 60" 
print np.percentile(b,60)
print "percentil 70" 
print np.percentile(b,70)
print "percentil 80" 
print np.percentile(b,80)
print "percentil 90" 
print np.percentile(b,90)
print "percentil 100" 
print np.percentile(b,100)
import matplotlib.pyplot as plt
print "bien 10 ter"
plt.hist(b, bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

plt.show()
        


    

#coords = np.vstack((lasnosuelofile.x, lasnosuelofile.y, lasnosuelofile.z)).transpose()
#print coords
"""for ex,ey,ez in np.nditer([x.las,y.las,z.las]):
    print ex"""
    
print "biene 11"

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def funcionchorra():    
        print "pasa"
    
 
#ejecuta la funcion que busca los archivos las y laz y a suvez ejecuta la funcion exprimelidar que hace el analisis de la cuadricula
buscalidaryejecuta(carpeta, crecimiento, fccbaja, fccterrazas, fccmedia, fccalta, hmontebravoe, hmontebravo, hselvicolas, hclaras, hclaras2, hbcminima, hbcdesarrollado, rcclaras, rcextremo, longitudcopaminima, crecimientofcc)

#defino una funcion que une en una capa el resultado de todas las hojas
def juntoshapes(busca,salida):
    files=glob.glob(busca)
    out=os.path.join(carpeta,salida+".shp")
    entrada=";".join(files)
    if len(files)>100:
        lista1=files[:len(files)/2]
        lista2=files[len(files)/2:]
        out=os.path.join(carpeta,salida+"1.shp")
        entrada=";".join(lista1)
        processing.runalg('saga:mergelayers',entrada,True,True,out)
        out=os.path.join(carpeta,salida+"2.shp")
        entrada=";".join(lista2)
        processing.runalg('saga:mergelayers',entrada,True,True,out)
    elif len(files) >1 and len(files) <=100:
        processing.runalg('saga:mergelayers',entrada,True,True,out)
    elif len(files) ==1:
        processing.runalg("qgis:saveselectedfeatures",files[0],out)
    else:
        pass
    del(out)
    del(entrada)
    del(files)
    
#uno en una capa todas las hojas de claras, regeneracion, resalveo y teselas
juntoshapes(os.path.join(carpeta,"p","*clara3.shp"),"Clara_merged")
juntoshapes(os.path.join(carpeta,"p","*regeneracion3.shp"),"Regeneracion_merged")
juntoshapes(os.path.join(carpeta,"p","*resalveo3.shp"),"Resalveo_merged")
juntoshapes(os.path.join(carpeta,"p","*suma.shp"),"Teselas_merged")

#elimino las capas que he cargado durante el proceso
capas =QgsMapLayerRegistry.instance().mapLayers()
for capa in capas:
    if capa not in capasoriginales:
        QgsMapLayerRegistry.instance().removeMapLayers( [capa] )
del(capas)
        
#cargo las capas finales
teselas=QgsVectorLayer(os.path.join(carpeta,'Teselas_merged.shp'),"Teselas","ogr")
teselas1=QgsVectorLayer(os.path.join(carpeta,'Teselas_merged_proyectado1.shp'),"Teselas Proyectado1","ogr")
teselas2=QgsVectorLayer(os.path.join(carpeta,'Teselas_merged_proyectado2.shp'),"Teselas Proyectado2","ogr")
clara=QgsVectorLayer(os.path.join(carpeta,'Clara_merged.shp'),"Clara","ogr")
regeneracion=QgsVectorLayer(os.path.join(carpeta,'Regeneracion_merged.shp'),"Regeneracion","ogr")
resalveo=QgsVectorLayer(os.path.join(carpeta,'Resalveo_merged.shp'),"Resalveo","ogr")

#aplico simbologia a estas capas, si existen
try:
    symbolsclara=clara.rendererV2().symbols()
    sym=symbolsclara[0]
    sym.setColor(QColor.fromRgb(255,0,0))
    QgsMapLayerRegistry.instance().addMapLayer(clara)
except: 
  pass

try:
    symbolsregeneracion=regeneracion.rendererV2().symbols()
    sym=symbolsregeneracion[0]
    sym.setColor(QColor.fromRgb(0,255,0))
    QgsMapLayerRegistry.instance().addMapLayer(regeneracion)
except: 
  pass

try:
    symbolsresalveo=resalveo.rendererV2().symbols()
    sym=symbolsresalveo[0]
    sym.setColor(QColor.fromRgb(0,0,255))
    QgsMapLayerRegistry.instance().addMapLayer(resalveo)
except: 
  pass

coloresteselas={"1":("solid","255,255,204,255","Raso o Regenerado","001"),"2":("solid","255,255,0,255","Menor (Monte Bravo)","002"),"3":("vertical","255,192,0,255","Poda Baja (y Clareo) en Bajo Latizal (Posibilidad si C elevada)","004"),"4":("solid","255,204,153,255","Bajo Latizal Desarrollado","005"),"51":("b_diagonal","255,0,255,255","Resalveo en Latizal poco desarrollado","006"),"52":("f_diagonal","255,0,0,255","Resalveo en Latizal","007"),"61":("solid","255,153,255,255","Latizal poco desarrollado Tratado","008"),"62":("solid","255,124,128,255","Latizal Tratado","009"),"7":("solid","204,255,153,255","Alto Latizal Claro","010"),"81":("b_diagonal","146,208,80,255","Poda Alta y Clara Suave en Latizal","011"),"82":("b_diagonal","51,204,204,255","Poda Alta y Clara Suave en Monte Desarrollado","015"),"9":("f_diagonal","0,176,80,255","Primera Clara y Poda Alta","012"),"10":("solid","102,255,153,255","Alto Latizal Aclarado","013"),"111":("solid","102,255,255,255","Fustal Claro","014"),"112":("solid","139,139,232,255","Fustal Maduro Claro","018"),"121":("f_diagonal","0,176,255,240","Clara en Fustal","016"),"122":("b_diagonal","65,51,162,255","Clara en Fustal Maduro","019"),"13":("cross","0,112,192,255","Clara Urgente en Fustal Maduro","020"),"141":("solid","204,236,255,255","Fustal Aclarado","017"),"142":("solid","166,166,207,255","Fustal Maduro Aclarado","021"),"15":("horizontal","112,48,160,255","Posibilidad de Regeneracion","022"),"17":("solid","orange","Bajo Latizal No Concurrente o Latizal Encinar no Denso","003")}

#ordeno los elementos de teselas
ordenados=coloresteselas.items()
ordenados.sort(key=lambda clave: str(clave[1][3]))

categorias=[]

for clase,(relleno,color, etiqueta,orden) in ordenados:    
    props={'style':relleno, 'color':color, 'style_border':'no'}
    sym=QgsFillSymbolV2.createSimple(props)
    categoria=QgsRendererCategoryV2(clase,sym,etiqueta)
    categorias.append(categoria)

field="DN"
renderer=QgsCategorizedSymbolRendererV2(field,categorias)
teselas.setRendererV2(renderer)
QgsMapLayerRegistry.instance().addMapLayer(teselas)

categorias1=[]
for clase,(relleno,color, etiqueta,orden) in ordenados:    
    props={'style':relleno, 'color':color, 'style_border':'no'}
    sym=QgsFillSymbolV2.createSimple(props)
    categoria1=QgsRendererCategoryV2(clase,sym,etiqueta)
    categorias1.append(categoria1)

field="DN"
renderer=QgsCategorizedSymbolRendererV2(field,categorias1)
teselas1.setRendererV2(renderer)
QgsMapLayerRegistry.instance().addMapLayer(teselas1)

categorias2=[]
for clase,(relleno,color, etiqueta,orden) in ordenados:    
    props={'style':relleno, 'color':color, 'style_border':'no'}
    sym=QgsFillSymbolV2.createSimple(props)
    categoria2=QgsRendererCategoryV2(clase,sym,etiqueta)
    categorias2.append(categoria2)

field="DN"
renderer=QgsCategorizedSymbolRendererV2(field,categorias2)
teselas2.setRendererV2(renderer)
QgsMapLayerRegistry.instance().addMapLayer(teselas2)

#repinto todo refrescando la vista
canvas.freeze(False)
canvas.refresh()
