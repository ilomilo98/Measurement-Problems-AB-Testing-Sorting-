############################################
# SORTING PRODUCTS
############################################
# Rating Products : puan hesaplama ürün puanlama
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

# Sorting Products : nasıl sıralarız? puana mı bakcaz en çok satana mı bakcaz?
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler

# Sorting Reviews : yoruma göre sıralama faydalı bulan insanlar..
# - Score Up-Down Diff
# - Average rating
# - Wilson Lower Bound Score binary ifade edilmeli 2li interaction sıralama.. Binary interactionları sıralamak için kullanılır.


import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(".../amazon_review.csv")
df.head()

df.asin.value_counts()
df.describe([0.01, 0.05, 0.1, 0.90, 0.95, 0.99])
df["overall"].mean() #puanların ortalaması :  4.587589013224822

########Average Rating’i güncel yorumlara göre hesaplamak ve var olan average rating ile kıyaslamak#######
# Rating Products-- olası faktörleri göz önünde bulundurarak ağırlıklı ürün puanlama---zamana göre hassaslaştırmalıyız. Çünkü İlgili olan müşteri memnuiyet trendi kaçırılabilir.
#Çok uzun zamandır var olan ve eskiden çok beğenilen ürünse ama son zamanda üretici tarafından ilgi düştüyse bu puanlara yansımıcaktır.
#bunun en iyisi time based ağırlandırma.. Zamana göre ağırlık vercez.

df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean() * 28 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() * 27 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() * 23 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean() * 22 / 100

# 4.596237959128027... Önceden ortalama 4.5875 iken şimdi 4.59623 oldu.
###################################################
# Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyip sıralamayı wlb yöntemine göre yapalım.
# Score = (up ratings) − (down ratings)
# elimizde down rating yok. Total vote ve helpful bilgisi var. Totalden helpful'u çıkarırsak helpful olmayanı buluruz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

def score_up_down_diff(up, down):
    return up - down

# Score = Average rating = (up ratings) / (down ratings)
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

# Wilson Lower Bound Score
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)
