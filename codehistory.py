import numpy as np
def veriyioku2():
    veri=open("history.txt","r").read()
    a=veri.replace("WARNING:tensorflow:Can save best model only with val_acc available, skipping."," ")
    b=a.replace("val_accuracy","Validation Accuracy ")
    veriler=b.split("\n")  #veri değişkenindeki \n lere göre parçalayıp
    textfile = open("deneme.txt", "w")
    textfile2 = open("sonucCsv.txt", "w")
    sonucCsv = ["EPOCH\tACC\tVAL_ACC"]
    textfile2.write(sonucCsv[0] + "\n" )
    
    counter = 1
    for index,satir in enumerate(veriler):   #dizinin boyu kadar döngü döndürür
     if(satir.startswith("352/352 [==============================]")):     #yandaki ile başlayan satırların hepsinde
         veriler[index] = satir[len("352/352 [==============================] - 121s 343ms/step - loss: 0.0335 - accuracy: "):-1].title()
         veriler[index] = veriler[index] [0: len("0.0000 - "):] + veriler[index] [len("0.0000 - Val_Loss: 0.0000 - Validation Accuracy : ")::]
         veriler[index]= veriler[index].replace(" - ","\t")
         sonucCsv.append(veriler[index])
         textfile2.write( str(counter) + "\t" + sonucCsv[counter] + "\n" )
         counter+=1
    
     textfile.write(veriler[index] + "\n")
     print(veriler[index])
   
    textfile2.close()
    textfile.close()


veriyioku2()

