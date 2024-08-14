#Bu örnekte Kaggle'da bulunan IRIS veri setini kullanacağız.
#IRIS veri seti 3 türe ait 50'şer örnekten 150 örnek bulunan veri setidir.
#Her bir veri örneği için 4 farklı özellik tanımlanmıştır: taç yaprak uzunluğu, taç yaprak genişliği
#çanak yaprak genişliği ve çanak yaprak uzunluğu.
#Şimdi gerekli kütüphanelerimizi içe aktaralım.
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#Csv dosyasındaki verilerimizi dataframe'e aktaralım ve ekrana ilk 5 veriyi yazdıralım.
url = "pca_iris.data"
df = pd.read_csv(url, names=["sepal length", "sepal width", "petal length", "petal width", "target"])
#print(df.head())


#Özelliklerimizi sonuç sütunundan ayırıyoruz.
features = ["sepal length", "sepal width", "petal length", "petal width"]
x = df[features]
y = df[["target"]]


#Şu anda dataframe'de bulunan değerleimizin bulunduğu aralıklar farklı. PCA algoritması bu verilerle
#yanlış sonuçlar elde edecektir. Bunu önlemek için bütün değerlerini standarize ediyoruz.
x = StandardScaler().fit_transform(x)
#print(x)


#Şimdi gelelim veri setimizin boyutunu 4'ten 2'ye indirmeye.
#Burada PCA algoritmasını kullanacağız.
pca = PCA(n_components=2) #İndirilmek istenen boyut sayısı
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=["principal component 1", "principal component 2"])
#print(principalDf.head())


#Şimdi de son olarak oluşturduğumuz principalDf dataframe'mine target sütununu ekliyoruz.
final_dataframe = pd.concat([principalDf, df[["target"]]], axis=1)
#print(final_dataframe.head())


#Oluşturduğumuz dataframe'i basit olarak görselleştirelim.
dfsetosa = final_dataframe[df.target == "Iris-setosa"]
dfvirginica = final_dataframe[df.target == "Iris-virginica"]
dfversicolor = final_dataframe[df.target == "Iris-versicolor"]
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.scatter(dfsetosa["principal component 1"], dfsetosa["principal component 2"], color="green")
plt.scatter(dfvirginica["principal component 1"], dfvirginica["principal component 2"], color="red")
plt.scatter(dfversicolor["principal component 1"], dfversicolor["principal component 2"], color="blue")
#plt.savefig("sonuc1.png", dpi=300)
#plt.show()


#Verilerimizi daha profesyonel şekilde görselleştirelim.
dizi_iris = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
dizi_colors = ["g", "r", "b"]
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

for iris, color in zip(dizi_iris, dizi_colors):
    dftemp = final_dataframe[df.target == iris]
    plt.scatter(dftemp["principal component 1"], dftemp["principal component 2"], color=color)

#plt.savefig("sonuc2.png", dpi=300)
#plt.show()


#Şimdi veri setindeki farklılıkların ne ölçüde korunduğuna bakalım.
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
#Toplamda %95 oranında veri setim korunmuştur.