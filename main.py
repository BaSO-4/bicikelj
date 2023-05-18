import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

station_names = "PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE,POGAČARJEV TRG-TRŽNICA,KONGRESNI TRG-ŠUBIČEVA ULICA,CANKARJEVA UL.-NAMA,BREG,GRUDNOVO NABREŽJE-KARLOVŠKA C.,MIKLOŠIČEV PARK,BAVARSKI DVOR,TRG OF-KOLODVORSKA UL.,MASARYKOVA DDC,VILHARJEVA CESTA,PARK NAVJE-ŽELEZNA CESTA,TRG MDB,PARKIRIŠČE NUK 2-FF,AMBROŽEV TRG,GH ŠENTPETER-NJEGOŠEVA C.,ILIRSKA ULICA,TRŽAŠKA C.-ILIRIJA,TIVOLI,STARA CERKEV,KINO ŠIŠKA,ŠPICA,BARJANSKA C.-CENTER STAREJŠIH TRNOVO,ZALOŠKA C.-GRABLOVIČEVA C.,TRŽNICA MOSTE,ROŽNA DOLINA-ŠKRABČEVA UL.,DUNAJSKA C.-PS PETROL,PLEČNIKOV STADION,DUNAJSKA C.-PS MERCATOR,LIDL - VOJKOVA CESTA,ŠPORTNI CENTER STOŽICE,KOPRSKA ULICA,MERCATOR CENTER ŠIŠKA,CITYPARK,BTC CITY/DVORANA A,BTC CITY ATLANTIS,TRNOVO,P+R BARJE,P + R DOLGI MOST,BONIFACIJA,ANTONOV TRG,BRATOVŠEVA PLOŠČAD,BS4-STOŽICE,SAVSKO NASELJE 2-LINHARTOVA CESTA,SAVSKO NASELJE 1-ŠMARTINSKA CESTA,SITULA,ŠTEPANJSKO NASELJE 1-JAKČEVA ULICA,HOFER-KAJUHOVA,BRODARJEV TRG,PREGLOV TRG,LIDL-LITIJSKA CESTA,ŽIVALSKI VRT,CESTA NA ROŽNIK,ŠMARTINSKI PARK,POLJANSKA-POTOČNIKOVA,SREDNJA FRIZERSKA ŠOLA,POVŠETOVA-GRABLOVIČEVA,TRŽNICA KOSEZE,LIDL BEŽIGRAD,MERCATOR MARKET - CELOVŠKA C. 163,RAKOVNIK,ALEJA - CELOVŠKA CESTA,IKEA,KOPALIŠČE KOLEZIJA,VIŠKO POLJE,KOSEŠKI BAJER,DRAVLJE,ČRNUČE,STUDENEC,POLJE,ZALOG,LIDL - RUDNIK,PRUŠNIKOVA,POVŠETOVA - KAJUHOVA,SOSESKA NOVO BRDO,TEHNOLOŠKI PARK,VOJKOVA - GASILSKA BRIGADA,GERBIČEVA - ŠPORTNI PARK SVOBODA,DOLENJSKA C. - STRELIŠČE,ROŠKA - STRELIŠKA,LEK - VEROVŠKOVA,VOKA - SLOVENČEVA,SUPERNOVA LJUBLJANA - RUDNIK"
station_names = station_names.split(",")

def main():
    # read data from files
    train = pd.read_csv("bicikelj_train.csv")
    train["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in train["timestamp"].values]
    test = pd.read_csv("bicikelj_test.csv")
    test["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in test["timestamp"].values]

    vreme = pd.read_csv("vreme.csv")
    vreme = vreme.iloc[:, 2:]
    vreme[" valid"] = [pd.to_datetime(ts).tz_localize(None) for ts in vreme[" valid"].values]
    vreme["količina padavin [mm]"] = vreme["količina padavin [mm]"].interpolate(method="nearest")

    data = train.copy()
    data["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in data["timestamp"].values]

# modify data
    # a sploh rabm to al se avtomatsko nardi?
    data["light rain"] = 0 
    data["heavy rain"] = 0
    data["low humidity"] = 0
    data["high humidity"] = 0
    data["temp 10-20"] = 0
    data["temp 20-30"] = 0
    data["temp 30+"] = 0
    data["weekend"] = 0
    data["school holiday"] = 0
    data["7-9"] = 0
    data["9-12"] = 0
    data["12-15"] = 0
    data["15-18"] = 0
    data["18-24"] = 0
    for i, t in enumerate(data["timestamp"].values):
        # weather
        closest = np.argmin(np.abs(vreme[" valid"].values - t))
        data.loc[i, "light rain"] = 1 if (vreme.loc[closest, "količina padavin [mm]"] > 0 and vreme.loc[closest, "količina padavin [mm]"] <= 2.5) else 0
        data.loc[i, "heavy rain"] = 1 if vreme.loc[closest, "količina padavin [mm]"] > 2.5 else 0
        data.loc[i, "low humidity"] = 1 if vreme.loc[closest, "povp. rel. vla. [%]"] <= 50 else 0
        data.loc[i, "high humidity"] = 1 if vreme.loc[closest, "povp. rel. vla. [%]"] > 50 else 0
        data.loc[i, "temp 10-20"] = 1 if vreme.loc[closest, "povp. T [°C]"] <= 20 else 0
        data.loc[i, "temp 20-30"] = 1 if (vreme.loc[closest, "povp. T [°C]"] > 20 and vreme.loc[closest, "povp. T [°C]"] <= 30) else 0
        data.loc[i, "temp 20-30"] = 1 if vreme.loc[closest, "povp. T [°C]"] > 30 else 0
        
        # weekend
        data.loc[i, "weekend"] = 1 if t.weekday() >= 5 else 0

        # school holiday
        data.loc[i, "school holiday"] = 1 if (t.month == 8) else 0

        # time
        data.loc[i, "7-9"] = 1 if (t.hour >= 7 and t.hour < 9) else 0
        data.loc[i, "9-12"] = 1 if (t.hour >= 9 and t.hour < 12) else 0
        data.loc[i, "12-15"] = 1 if (t.hour >= 12 and t.hour < 15) else 0
        data.loc[i, "15-18"] = 1 if (t.hour >= 15 and t.hour < 18) else 0
        data.loc[i, "18-24"] = 1 if (t.hour >= 18 and t.hour < 24) else 0

        # bikes before
        # ne vem kako se odštejejo časi
        # data.loc[i, "bikes_30min"] = data.loc[i-30min, "bikes"]
        data.loc[i, "bikes_1h"] = data.loc[i - 1, "bikes"]
        # data.loc[i, "bikes_90min"] = data.loc[i-90min, "bikes"]
        data.loc[i, "bikes_2h"] = data.loc[i - 2, "bikes"]



    print(data.head())

    for station in data.columns:
        target = data[station]
        station_data1 = data.drop(columns=station_names.remove(station)) # temu damo kolesa od zdej nazaj 
        station_data2 = data.drop(columns=station_names.remove(station)) # temu damo kolesa od pred 1h nazaj 
        model_1h = LinearRegression()
        model_2h = LinearRegression()
        model_1h.fit(station_data1, target) 
        model_1h.fit(station_data2, target)
        
    # test.to_csv("closest_time.csv", sep=",", index=False)

if __name__ == '__main__':
    main()