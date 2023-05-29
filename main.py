import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

station_names = "PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE,POGAČARJEV TRG-TRŽNICA,KONGRESNI TRG-ŠUBIČEVA ULICA,CANKARJEVA UL.-NAMA,BREG,GRUDNOVO NABREŽJE-KARLOVŠKA C.,MIKLOŠIČEV PARK,BAVARSKI DVOR,TRG OF-KOLODVORSKA UL.,MASARYKOVA DDC,VILHARJEVA CESTA,PARK NAVJE-ŽELEZNA CESTA,TRG MDB,PARKIRIŠČE NUK 2-FF,AMBROŽEV TRG,GH ŠENTPETER-NJEGOŠEVA C.,ILIRSKA ULICA,TRŽAŠKA C.-ILIRIJA,TIVOLI,STARA CERKEV,KINO ŠIŠKA,ŠPICA,BARJANSKA C.-CENTER STAREJŠIH TRNOVO,ZALOŠKA C.-GRABLOVIČEVA C.,TRŽNICA MOSTE,ROŽNA DOLINA-ŠKRABČEVA UL.,DUNAJSKA C.-PS PETROL,PLEČNIKOV STADION,DUNAJSKA C.-PS MERCATOR,LIDL - VOJKOVA CESTA,ŠPORTNI CENTER STOŽICE,KOPRSKA ULICA,MERCATOR CENTER ŠIŠKA,CITYPARK,BTC CITY/DVORANA A,BTC CITY ATLANTIS,TRNOVO,P+R BARJE,P + R DOLGI MOST,BONIFACIJA,ANTONOV TRG,BRATOVŠEVA PLOŠČAD,BS4-STOŽICE,SAVSKO NASELJE 2-LINHARTOVA CESTA,SAVSKO NASELJE 1-ŠMARTINSKA CESTA,SITULA,ŠTEPANJSKO NASELJE 1-JAKČEVA ULICA,HOFER-KAJUHOVA,BRODARJEV TRG,PREGLOV TRG,LIDL-LITIJSKA CESTA,ŽIVALSKI VRT,CESTA NA ROŽNIK,ŠMARTINSKI PARK,POLJANSKA-POTOČNIKOVA,SREDNJA FRIZERSKA ŠOLA,POVŠETOVA-GRABLOVIČEVA,TRŽNICA KOSEZE,LIDL BEŽIGRAD,MERCATOR MARKET - CELOVŠKA C. 163,RAKOVNIK,ALEJA - CELOVŠKA CESTA,IKEA,KOPALIŠČE KOLEZIJA,VIŠKO POLJE,KOSEŠKI BAJER,DRAVLJE,ČRNUČE,STUDENEC,POLJE,ZALOG,LIDL - RUDNIK,PRUŠNIKOVA,POVŠETOVA - KAJUHOVA,SOSESKA NOVO BRDO,TEHNOLOŠKI PARK,VOJKOVA - GASILSKA BRIGADA,GERBIČEVA - ŠPORTNI PARK SVOBODA,DOLENJSKA C. - STRELIŠČE,ROŠKA - STRELIŠKA,LEK - VEROVŠKOVA,VOKA - SLOVENČEVA,SUPERNOVA LJUBLJANA - RUDNIK"
station_names = station_names.split(",")
time_intervals = ["00-03","03-07","07-09","09-12","12-15","15-17","17-20","20-24"]

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
    endgame = test.copy()


    # preprocess data

    # Weather conditions
    closest = np.argmin(np.abs(vreme[" valid"].values[:, None] - data["timestamp"].values), axis=0)
    closest_test = np.argmin(np.abs(vreme[" valid"].values[:, None] - endgame["timestamp"].values), axis=0)
    data["light rain"] = np.where((vreme.loc[closest, "količina padavin [mm]"].values > 0) & (vreme.loc[closest, "količina padavin [mm]"].values <= 2.5), 1, 0)
    test["light rain"] = np.where((vreme.loc[closest_test, "količina padavin [mm]"].values > 0) & (vreme.loc[closest_test, "količina padavin [mm]"].values <= 2.5), 1, 0)
    data["heavy rain"] = np.where(vreme.loc[closest, "količina padavin [mm]"].values > 2.5, 1, 0)
    test["heavy rain"] = np.where(vreme.loc[closest_test, "količina padavin [mm]"].values > 2.5, 1, 0)
    data["low humidity"] = np.where(vreme.loc[closest, "povp. rel. vla. [%]"].values <= 50, 1, 0)
    test["low humidity"] = np.where(vreme.loc[closest_test, "povp. rel. vla. [%]"].values <= 50, 1, 0)
    data["high humidity"] = np.where(vreme.loc[closest, "povp. rel. vla. [%]"].values > 50, 1, 0)
    test["high humidity"] = np.where(vreme.loc[closest_test, "povp. rel. vla. [%]"].values > 50, 1, 0)
    data["temp < 20"] = np.where(vreme.loc[closest, "povp. T [°C]"].values <= 20, 1, 0)
    test["temp < 20"] = np.where(vreme.loc[closest_test, "povp. T [°C]"].values <= 20, 1, 0)
    data["temp 20-30"] = np.where((vreme.loc[closest, "povp. T [°C]"].values > 20) & (vreme.loc[closest, "povp. T [°C]"].values <= 30), 1, 0)
    test["temp 20-30"] = np.where((vreme.loc[closest_test, "povp. T [°C]"].values > 20) & (vreme.loc[closest_test, "povp. T [°C]"].values <= 30), 1, 0)
    data["temp > 30"] = np.where(vreme.loc[closest, "povp. T [°C]"].values > 30, 1, 0)
    test["temp > 30"] = np.where(vreme.loc[closest_test, "povp. T [°C]"].values > 30, 1, 0)
    
    # special lines to check if there was rain on that day (to izboljša rezultate za 0.002 na testni, pustim noter ker verjetno pomaga)
    vreme["date"] = pd.to_datetime(vreme[" valid"]).dt.date
    vreme["date"] = pd.to_datetime(vreme["date"]).dt.tz_localize(None)
    vreme["rain that date"] = np.where(vreme["količina padavin [mm]"] > 0, 1, 0)
    vreme = vreme.drop_duplicates(subset="date", keep="first")
    rain = vreme[["date", "rain that date"]]
    data["date"] = pd.to_datetime(data["timestamp"]).dt.date
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    test["date"] = pd.to_datetime(test["timestamp"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.tz_localize(None)
    data = pd.merge_asof(data, rain[["date", "rain that date"]], on="date", direction="nearest")
    test = pd.merge_asof(test, rain[["date", "rain that date"]], on="date", direction="nearest")
    data = data.drop(columns=["date"])
    test = test.drop(columns=["date"])

    # Weekend
    data["saturday"] = np.where(data["timestamp"].dt.weekday == 5, 1, 0)
    test["saturday"] = np.where(endgame["timestamp"].dt.weekday == 5, 1, 0)
    data["sunday"] = np.where(data["timestamp"].dt.weekday == 6, 1, 0)
    test["sunday"] = np.where(endgame["timestamp"].dt.weekday == 6, 1, 0)

    # School holiday
    data["school holiday"] = np.where(data["timestamp"].dt.month == 8, 1, 0)
    test["school holiday"] = np.where(endgame["timestamp"].dt.month == 8, 1, 0)

    # student exam period
    # data["exam period"] = np.where(data["timestamp"].dt.date >= pd.to_datetime("15.9.2022").date(), 1, 0)
    # test["exam period"] = np.where(endgame["timestamp"].dt.date >= pd.to_datetime("15.9.2022").date(), 1, 0)
         
    # Time of day
    data["00-03"] = np.where((data["timestamp"].dt.hour >= 0) & (data["timestamp"].dt.hour < 3), 1, 0)
    test["00-03"] = np.where((endgame["timestamp"].dt.hour >= 0) & (endgame["timestamp"].dt.hour < 3), 1, 0)
    data["03-07"] = np.where((data["timestamp"].dt.hour >= 3) & (data["timestamp"].dt.hour < 7), 1, 0)
    test["03-07"] = np.where((endgame["timestamp"].dt.hour >= 3) & (endgame["timestamp"].dt.hour < 7), 1, 0)
    data["07-09"] = np.where((data["timestamp"].dt.hour >= 7) & (data["timestamp"].dt.hour < 9), 1, 0)
    test["07-09"] = np.where((endgame["timestamp"].dt.hour >= 7) & (endgame["timestamp"].dt.hour < 9), 1, 0)
    data["09-12"] = np.where((data["timestamp"].dt.hour >= 9) & (data["timestamp"].dt.hour < 12), 1, 0)
    test["09-12"] = np.where((endgame["timestamp"].dt.hour >= 9) & (endgame["timestamp"].dt.hour < 12), 1, 0)
    data["12-15"] = np.where((data["timestamp"].dt.hour >= 12) & (data["timestamp"].dt.hour < 15), 1, 0)
    test["12-15"] = np.where((endgame["timestamp"].dt.hour >= 12) & (endgame["timestamp"].dt.hour < 15), 1, 0)
    data["15-17"] = np.where((data["timestamp"].dt.hour >= 15) & (data["timestamp"].dt.hour < 17), 1, 0)
    test["15-17"] = np.where((endgame["timestamp"].dt.hour >= 15) & (endgame["timestamp"].dt.hour < 17), 1, 0)
    data["17-20"] = np.where((data["timestamp"].dt.hour >= 17) & (data["timestamp"].dt.hour < 20), 1, 0)
    test["17-20"] = np.where((endgame["timestamp"].dt.hour >= 17) & (endgame["timestamp"].dt.hour < 20), 1, 0)
    data["20-24"] = np.where((data["timestamp"].dt.hour >= 20) & (data["timestamp"].dt.hour < 24), 1, 0)
    test["20-24"] = np.where((endgame["timestamp"].dt.hour >= 20) & (endgame["timestamp"].dt.hour < 24), 1, 0)

    # data["day of the week"] = data["timestamp"].dt.weekday
    # test["day of the week"] = endgame["timestamp"].dt.weekday

    mins_60_before = data.copy()
    mins_60_before["timestamp"] = data["timestamp"] + pd.Timedelta(minutes=60)
    mins_90_before = data.copy()
    mins_90_before["timestamp"] = data["timestamp"] + pd.Timedelta(minutes=90)
    mins_120_before = data.copy()
    mins_120_before["timestamp"] = data["timestamp"] + pd.Timedelta(minutes=120)
    mins_180_before = data.copy()
    mins_180_before["timestamp"] = data["timestamp"] + pd.Timedelta(minutes=180)
    mins_210_before = data.copy()
    mins_210_before["timestamp"] = data["timestamp"] + pd.Timedelta(minutes=210)
    mins_240_before = data.copy()
    mins_240_before["timestamp"] = data["timestamp"] + pd.Timedelta(minutes=240)
    mins_270_before = data.copy()
    mins_270_before["timestamp"] = data["timestamp"] + pd.Timedelta(minutes=270)

    test.drop(columns=station_names, inplace=True)


    # build 2 models for each station
    for station in station_names:
        target = data[station]

        # drop station names
        # _station_names = station_names.copy()
        # _station_names.remove(station)
        # station_data = data.drop(columns=_station_names)

        # # create a new dataframe with averages for each day of the week at each time interval
        # averages = pd.DataFrame(columns=time_intervals)
        # for interval in time_intervals:
        #     filtered_data = station_data[station_data[interval] == 1]
        #     average = filtered_data.groupby("day of the week")[station].mean()
        #     # averages[interval] = average
        #     averages[interval] = np.round(average)

        # # insert averages into station_data
        # valid_intervals = station_data[time_intervals].eq(1)
        # valid_intervals_index = valid_intervals.apply(lambda x: x.index[x].tolist(), axis=1)
        # station_data["average at interval and weekday"] = averages.lookup(station_data["day of the week"], valid_intervals_index.str[0])
        # # insert averages into test_data
        # valid_intervals = test[time_intervals].eq(1)
        # valid_intervals_index = valid_intervals.apply(lambda x: x.index[x].tolist(), axis=1)
        # test["average at interval and weekday"] = averages.lookup(test["day of the week"], valid_intervals_index.str[0])


        # station_data1 = station_data.drop(columns=["day of the week", station])
        # station_data2 = station_data.drop(columns=["day of the week", station])
        station_data1 = data.drop(columns=station_names)
        station_data2 = data.drop(columns=station_names)

        # how many bikes were there 60, 90, 120, 180, 210, 240, 270 minutes before 
        station_data1 = pd.merge_asof(station_data1, mins_60_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data1.rename(columns={station: "60 min before"}, inplace=True)
        station_data1 = pd.merge_asof(station_data1, mins_90_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data1.rename(columns={station: "90 min before"}, inplace=True)
        station_data1 = pd.merge_asof(station_data1, mins_120_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data1.rename(columns={station: "120 min before"}, inplace=True)
        station_data1 = pd.merge_asof(station_data1, mins_180_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data1.rename(columns={station: "180 min before"}, inplace=True)
        station_data1 = pd.merge_asof(station_data1, mins_210_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data1.rename(columns={station: "210 min before"}, inplace=True)
        station_data1.drop(columns=["timestamp"], inplace=True)

        station_data2 = pd.merge_asof(station_data2, mins_120_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data2.rename(columns={station: "120 min before"}, inplace=True)
        station_data2 = pd.merge_asof(station_data2, mins_180_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data2.rename(columns={station: "180 min before"}, inplace=True)
        station_data2 = pd.merge_asof(station_data2, mins_210_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data2.rename(columns={station: "210 min before"}, inplace=True)
        station_data2 = pd.merge_asof(station_data2, mins_240_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data2.rename(columns={station: "240 min before"}, inplace=True)
        station_data2 = pd.merge_asof(station_data2, mins_270_before[["timestamp", station]], on="timestamp", direction="nearest")
        station_data2.rename(columns={station: "270 min before"}, inplace=True)
        station_data2.drop(columns=["timestamp"], inplace=True)        

        # test_1h = test.drop(columns=["day of the week"])
        # test_2h = test.drop(columns=["day of the week"])
        test_1h = test.copy()
        test_2h = test.copy()

        test_1h = pd.merge_asof(test_1h, mins_60_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_1h.rename(columns={station: "60 min before"}, inplace=True)
        test_1h = pd.merge_asof(test_1h, mins_90_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_1h.rename(columns={station: "90 min before"}, inplace=True)
        test_1h = pd.merge_asof(test_1h, mins_120_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_1h.rename(columns={station: "120 min before"}, inplace=True)
        test_1h = pd.merge_asof(test_1h, mins_180_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_1h.rename(columns={station: "180 min before"}, inplace=True)
        test_1h = pd.merge_asof(test_1h, mins_210_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_1h.rename(columns={station: "210 min before"}, inplace=True)

        test_2h = pd.merge_asof(test_2h, mins_120_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_2h.rename(columns={station: "120 min before"}, inplace=True)
        test_2h = pd.merge_asof(test_2h, mins_180_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_2h.rename(columns={station: "180 min before"}, inplace=True)
        test_2h = pd.merge_asof(test_2h, mins_210_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_2h.rename(columns={station: "210 min before"}, inplace=True)
        test_2h = pd.merge_asof(test_2h, mins_240_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_2h.rename(columns={station: "240 min before"}, inplace=True)
        test_2h = pd.merge_asof(test_2h, mins_270_before[["timestamp", station]], on="timestamp", direction="nearest")
        test_2h.rename(columns={station: "270 min before"}, inplace=True)

        test_1h.drop(columns=["timestamp"], inplace=True)
        test_2h.drop(columns=["timestamp"], inplace=True)
        print(station_data1.keys())
        # model
        # model_1h = LinearRegression()
        # model_2h = LinearRegression()
        model_1h = GradientBoostingRegressor(loss="absolute_error")
        model_2h = GradientBoostingRegressor(loss="absolute_error")
        # model_1h = RandomForestRegressor()
        # model_2h = RandomForestRegressor()
        # model_1h = SVR()
        # model_2h = SVR()
        # model_1h = DecisionTreeRegressor()
        # model_2h = DecisionTreeRegressor()


        model_1h.fit(station_data1, target) 
        model_2h.fit(station_data2, target)

        model_1h_pred = model_1h.predict(test_1h)
        model_2h_pred = model_2h.predict(test_2h)

        # round to an integer
        model_1h_pred = np.round(model_1h_pred)
        model_2h_pred = np.round(model_2h_pred)

        # insert predictions into endgame
        endgame.loc[::2, station] = model_1h_pred[::2]
        endgame.loc[1::2, station] = model_2h_pred[1::2]
    

    # save endgame into a new csv file
    endgame.to_csv("gradient_boosting_absolute_error.csv", sep=",", index=False)
        

if __name__ == '__main__':
    main()