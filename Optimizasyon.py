import os, math, time, numpy, pandas, random, matplotlib
import numpy.random as nrand
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize

class Grid:
    def __init__(self, height, width, path, rand_test=True):
        """
        Bu metod Grid objesi oluşturulurken çalıştırılır. 
        Grid temelinde 2 boyutlu bir Datum nesnesi.
        :param height: grid yüksekliği
        :param width: grid genişliği
        """
        self.path = path
        # Grid boyutunu kaydet
        self.dim = numpy.array([height, width])
        # Datum tipinde bir numpy matrisi oluştur
        self.grid = numpy.empty((height, width), dtype=Datum)
        if rand_test:
            # Gridi rastgele doldur
            self.rand_grid(0.25)
        #plt.ion()
        #plt.figure(figsize=(10, 10))
        self.max_d = 0.001

    def rand_grid(self, sparse):
        """
        Rastgele değerlerle grid oluşturur
        :param sparse: gridin doldurulma oranı
        """
        for y in range(self.dim[0]):
            for x in range(self.dim[1]):
                if random.random() <= sparse:
                    r = random.randint(0, 1)
                    if r == 0:
                        self.grid[y][x] = Datum(nrand.normal(5, 0.25, 10))
                    elif r == 1:
                        self.grid[y][x] = Datum(nrand.normal(-5, 0.25, 10))

    def matrix_grid(self):
        """
        Bu metod gridi görselleştirebilmek için 2 boyuta sıkıştırır.
        :return: grid matrisini döndürür
        """
        matrix = numpy.empty((self.dim[0], self.dim[1]))
        matrix.fill(0)
        for y in range(self.dim[0]):
            for x in range(self.dim[1]):
                if self.grid[y][x] is not None:
                    matrix[y][x] = self.get_grid()[y][x].condense()
        return matrix

    def plot_grid(self, name="", save_figure=True):
        """
        Gridin 2 boyutlu gösterimi
        :param name: kaydedilecek resmin adı
        :return:
        """
        plt.matshow(self.matrix_grid(), cmap="RdBu", fignum=0) # 
        # Resmi kaydetme opsiyonu
        if save_figure:
            plt.savefig(self.path + name + '.png')
        #plt.draw()

    def get_grid(self):
        return self.grid

    def get_probability(self, d, y, x, n, c):
        """
        Girilen Datumun alınma ve bırakılmak olasılığını hesaplar, d
        :param d: Datum
        :param x: Datumun/Taşıyan karıncanın x konum değeri
        :param y: Datumun/Taşıyan karıncanın y konum değeri
        :param n: Komşu fonksiyonun boyutu
        :param c: Yakınlık kontrolü için sabit
        :return: Olasılığı döndürür
        """
        # x ve y konumlarından başla
        y_s = y - n
        x_s = x - n
        total = 0.0
        # Herbir komşuyu
        for i in range((n*2)+1):
            xi = (x_s + i) % self.dim[0]
            for j in range((n*2)+1):
                # Bir komşuya bakıyorsak
                if j != x and i != y:
                    yj = (y_s + j) % self.dim[1]
                    # Komşuyu al, o
                    o = self.grid[xi][yj]
                    # o nun x e benzerlik değerini al
                    if o is not None:
                        s = d.similarity(o)
                        total += s
        # Yoğunluğu o ana kadarki mesafeye göre normalize et
        md = total / (math.pow((n*2)+1, 2) - 1)
        if md > self.max_d:
            self.max_d = md
        density = total / (self.max_d * (math.pow((n*2)+1, 2) - 1))
        density = max(min(density, 1), 0)
        t = math.exp(-c * density)
        probability = (1-t)/(1+t)
        return probability

class Ant:
    def __init__(self, y, x, grid):
        """
        Hafızasız basit bir karınca nesnesi oluşturur
        :param y: y başlangıç konumu
        :param x: x başlangıç konumu
        :param grid: grid referansı
        """
        self.loc = numpy.array([y, x])
        self.carrying = grid.get_grid()[y][x]
        self.grid = grid

    def move(self, n, c):
        """
        Karıncaları gridde gezdirir
        :param step_size(n, c): Adım mesafesi
        """
        step_size = random.randint(1, 25)
        # Karınca konumuna (-1,+1) * step_size değerinde vektör ekle
        self.loc += nrand.randint(-1 * step_size, 1 * step_size, 2)
        # Taşma olmaması için Grid boyutuna göre modunu al
        self.loc = numpy.mod(self.loc, self.grid.dim)
        # O konumdaki nesneyi yakala
        o = self.grid.get_grid()[self.loc[0]][self.loc[1]]
        # Eğer hücre doluysa tekrar ilerle
        if o is not None:
            # Eğer karınca bir nesne taşımıyorsa
            if self.carrying is None:
                # Karıncanın nesneyi alıp almayacağını kontrol et
                if self.p_pick_up(n, c) >= random.random():
                    # Nesneyi yakala ve Gridden kaldır
                    self.carrying = o
                    self.grid.get_grid()[self.loc[0]][self.loc[1]] = None
                # Değilse tekrar ilerle
                else:
                    self.move(n, c)
            # Eğer bir nesne taşıyorsa ilerlemeye devam etsin
            else:
                self.move(n, c)
        # Eğer hücre boşsa
        else:
            if self.carrying is not None:
                # Karıncanın bırakıp bırakmayacağını kontrol et
                if self.p_drop(n, c) >= random.random():
                    # Boş konuma nesneyi bırak
                    self.grid.get_grid()[self.loc[0]][self.loc[1]] = self.carrying
                    self.carrying = None

    def p_pick_up(self, n, c):
        """
        Nesneyi yakalama olasılığını hesaplar
        :param n: Komşu boyutu
        :return: Yakalama olasılığı
        """
        ant = self.grid.get_grid()[self.loc[0]][self.loc[1]]
        return 1 - self.grid.get_probability(ant, self.loc[0], self.loc[1], n, c)

    def p_drop(self, n, c):
        """
        Nesneyi bırakma oranını hesaplar
        :return: Bırakma olasılığı
        """
        ant = self.carrying
        return self.grid.get_probability(ant, self.loc[0], self.loc[1], n, c)

class Datum:
    def __init__(self, data):
        """
        Datanum temel olarak N boyutlu bir vektör
        :param data: N boyutlu veri
        """
        self.data = data

    def similarity(self, datum):
        """
        2 Datanum arasındaki mesafenin kare-toplamını hesaplar
        :param datum: the other datum
        :return: sum squared distance
        """
        diff = numpy.abs(self.data - datum.data)
        return numpy.sum(diff**2)

    def condense(self):
        """
        Görselleştirmek için N Boyutlu vektörü 1 boyuta indirger
        :return: Vektörün 1 boyutlu gösterimi
        """
        return numpy.mean(self.data)

def optimize(height=50, width=50, ants=250, sims=500, n=5, c=5, freq=500, path="image"):
    
    # Gridi tanımla
    grid = Grid(height, width, path)
    # Karıncaları oluştur
    ant_agents = []
    for i in range(ants):
        ant = Ant(random.randint(0, height - 1), random.randint(0, width - 1), grid)
        ant_agents.append(ant)
        
    for i in range(sims):
        for ant in ant_agents:
            ant.move(n, c)
        # Her 10 adımda bir grid drumunu resim halinde kaydet
        if i%10 == 0:
            s = "img" + str(i).zfill(6)
            grid.plot_grid(s)

if __name__ == '__main__':
	DIR_NAME = "steps_1"
	optimize(path=DIR_NAME+"/")

