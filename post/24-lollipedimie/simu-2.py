import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

N = 1*13  # Wie viele sind es
anstckung = 5  # Ansteckungsschwelle
gesundung = 2  # Gesundungsschwelle
plottenbitte = True
repeat = 10

r = 2  # wieviele Runden/Tag
t = 3  # wieviele Tage

popmat = np.zeros((N, t+1))  # die Datenmatrix
people = np.zeros(N+2, )  # +2 um die Nachbarn zu modellieren


for nexp in range(repeat):
    # Initialisierung (hat wer 'ne 6)
    awrfln = randint(1, 7, size=(N, ))
    mitx = awrfln == 6

    people[1:-1] = mitx
    popmat[:, 0] = people[1:-1]

    for tag in range(t):

        # Nacht -- einige gesunden...
        awrfln = randint(1, 7, size=(N, ))
        xweg = awrfln <= gesundung
        xwegx = np.r_[False, xweg, False]
        people[xwegx] = 0

        # Ansteckung
        for runde in range(r):
            people[0] = people[-2]  # periodische Randbedingungen :)
            people[-1] = people[1]  # periodische Randbedingungen
            awrfln = randint(1, 7, size=(N, ))
            for who in range(1, N+1):
                if (awrfln[who-1] >= anstckung and
                        people[who-1] or people[who+1]):
                    people[who] = 1  # hat x bekommen

        # Wie siehts aus am Ende des Tages
        # plt.figure()
        popmat[:, tag+1] = people[1:-1]
        # plt.spy(popmat, marker='x')

        xweg = awrfln <= gesundung
        xwegx = np.r_[False, xweg, False]
        people[xwegx] = 0

    if plottenbitte and nexp == repeat-1:
        plt.figure(figsize=(4, 10))
        plt.xlabel('Tag')
        plt.ylabel('Proband:in')
        plt.spy(popmat, marker='.', aspect='auto')

    endx = int(popmat[:, -1].sum())

    print(f'Durchlauf {nexp+1:04d}: N={N} | Ansteckung:>={anstckung} | '
          f'Gesundung<={gesundung}: | ' +
          f'Am Ende hatten {endx} was: das entspricht {100*endx/N:.1f}%')

plt.show()
