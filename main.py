import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import urllib
import matplotlib.pyplot as plt
from scipy.misc import imread
import math
from scipy import interpolate, stats, special
MERCATOR_RANGE = 256

class G_Point :
    def __init__(self,x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "(%f, %f)" % (self.x, self.y)

    def __add__(self, other):
        return G_Point(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return self + (other*(-1))

    def __mul__(self, a):
        if type(a) == type(1.2) or type(a) == type(1):
            return G_Point(a*self.x, a*self.y)

    def tolist(self):
        return [self.x, self.y]

    def dot(self, other):
        return np.array(self.tolist()).dot(np.array(other.tolist()))

class G_LatLng :
    def __init__(self,lt, ln):
        self.lat = lt
        self.lng = ln

    def __repr__(self):
        return "(%f, %f)" % (self.lat, self.lng)

class BackgroundMap(object):
    def __init__(self, center):
        self.pixelOrigin_ =  G_Point( MERCATOR_RANGE / 2, MERCATOR_RANGE / 2)
        self.pixelsPerLonDegree_ = MERCATOR_RANGE / 360.0
        self.pixelsPerLonRadian_ = MERCATOR_RANGE / (2 * math.pi)
        self.center = center

        self.zoom = 12
        self.mapWidth = 640
        self.mapHeight = 640
        self.bounds = self.corners()
        self.NWmerc = None

    def latLngToMercator(self, latLng):
        #developers.google.com/maps/documentation/javascript/examples/map-coordinates
        #web mercator projection
        point = G_Point(0,0)
        origin = self.pixelOrigin_
        siny = math.sin(latLng.lat * math.pi/180.0)
        siny = min(max(siny, -0.9999), 0.9999)

        point.x = origin.x + latLng.lng * self.pixelsPerLonDegree_
        point.y = origin.y - (math.log((1 + siny) / (1 - siny)) / 2)* self.pixelsPerLonRadian_ 
        return point

    def mercatorToLatLng(self,point) :
        origin = self.pixelOrigin_
        lng = (point.x - origin.x) / self.pixelsPerLonDegree_
        latRadians = (point.y - origin.y) / -self.pixelsPerLonRadian_
        lat = (180/math.pi)*(2 * math.atan(math.exp(latRadians)) - math.pi / 2)
        return G_LatLng(lat, lng)

    def latlngToPixel(self, latlng):
        if self.NWmerc == None:
            NWLatLng = G_LatLng(self.bounds['N'], self.bounds['W'])
            self.NWmerc = self.latLngToMercator(NWLatLng)

        latlngmerc = self.latLngToMercator(latlng)

        diff = latlngmerc - self.NWmerc
        scale = float(2**self.zoom)
        latlngpx = diff*scale*2
        return G_Point(math.floor(latlngpx.x), math.floor(latlngpx.y))

    def pixelToLatLng(self, pixel):
        if self.NWmerc == None:
            NWLatLng = G_LatLng(self.bounds['N'], self.bounds['W'])
            self.NWmerc = self.latLngToMercator(NWLatLng)

        scale = float(2**self.zoom)
        latLngMerc = G_Point(0,0) 

        latLngMerc.x = pixel.x/(scale*2)
        latLngMerc.y = pixel.y/(scale*2)

        return self.mercatorToLatLng(latLngMerc + self.NWmerc) 

    def latLngDistance(self, A, B):
	#distance in km
	D_x = (A.lng - B.lng) * np.cos(B.lat * np.pi / 180) * 111.323
	D_y = (A.lat - B.lat) * 111.323
        return np.sqrt(D_x*D_x + D_y*D_y)

    def corners(self):
        scale = float(2**self.zoom)
        centerMercator = self.latLngToMercator(self.center)
        SWPoint = G_Point(centerMercator.x-(self.mapWidth/2)/scale, centerMercator.y+(self.mapHeight/2)/scale)
        SWLatLon = self.mercatorToLatLng(SWPoint)
        NEPoint = G_Point(centerMercator.x+(self.mapWidth/2)/scale, centerMercator.y-(self.mapHeight/2)/scale)
        NELatLon = self.mercatorToLatLng(NEPoint)
        return {'N' : NELatLon.lat,
                'E' : NELatLon.lng,
                'S' : SWLatLon.lat,
                'W' : SWLatLon.lng,}

    def get_map(self):
	c = {}
	c["lat"] = self.center.lat
	c["long"] = self.center.lng
	c["zoom"] = self.zoom
        c["mapWidth"] = self.mapWidth
        c["mapHeight"] = self.mapHeight
	s = "https://maps.googleapis.com/maps/api/staticmap?center={lat},{long}&zoom={zoom}&size={mapWidth}x{mapHeight}&scale=2".format(**c)

	#add gate
	s += "&markers=color:red|52.516288,13.377689"

	#add river
        spree = River(self).get_data("spree.dat") 
	s += "&path=color:blue|weight:5|" + spree

	#add satellite
	s += "&path=color:green|weight:5|52.590117,13.39915|52.437385,13.553989" 

	image = urllib.URLopener()
	image.retrieve(s, "map.png")

class DensityMap(object):
    def __init__(self):
        map_center = G_LatLng(52.5089253483, 13.4215381072)
        background = BackgroundMap(map_center)

	if not os.path.exists('map.png'):
            c = background.get_map()

        gate = G_LatLng(52.516288,13.377689)
        myGate = Gate(background, gate)
        #plot gate pdf
        plt.clf()
        plt.figure(frameon=False, figsize=(1280/72., 1280/72.), dpi=72)
        plt.axis('off')

        Z = self.main_loop(background, [myGate])
        plt.contourf(Z, cmap=plt.cm.bone, alpha=0.5)

        img = imread("map.png")
        plt.imshow(img,zorder=0)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.savefig("gate.png", dpi=72)

	S0 = G_LatLng(52.590117,13.39915)
	S1 = G_LatLng(52.437385,13.553989)
        sat = Satellite(background, S0, S1)
        #plot satellite pdf
        plt.clf()
        plt.figure(frameon=False, figsize=(1280/72., 1280/72.), dpi=72)
        plt.axis('off')

        Z = self.main_loop(background, [sat])
        plt.contourf(Z, cmap=plt.cm.bone, alpha=0.5)

        plt.imshow(img,zorder=0)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.savefig("sat.png", dpi=72)

        river = River(background, setup=True)
        #plot river pdf
        plt.clf()
        plt.figure(frameon=False, figsize=(1280/72., 1280/72.), dpi=72)
        plt.axis('off')

        Z = self.main_loop(background, [river])
        plt.contourf(Z, cmap=plt.cm.bone, alpha=0.5)

        plt.imshow(img,zorder=0)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.savefig("river.png", dpi=72)

        #calculate joint pdf on grid
        Z = self.main_loop(background, [myGate, sat, river], weights=[1/3., 1/3., 1/3.])

        #plot 1% and 5% errors in the analyst position
        def find_nearest_idx(array,value):
            return (np.abs(array-value)).argmin()
        a = sorted(Z.flatten())
        res = stats.cumfreq(a, numbins=25)
        x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
        ys = res.cumcount * (1/res.cumcount[-1])
        idx_95 = find_nearest_idx(ys, 0.95)
        idx_99 = find_nearest_idx(ys, 0.99)
        v95 = x[idx_95]
        v99 = x[idx_99]

        #plot cdf and estimate the 1% and 5% errors
        plt.clf()
        plt.figure(frameon=True, figsize=(600/72., 600/72.), dpi=72)
        plt.axis('on')

        plt.scatter(x, ys)
        plt.ylim([0.0,1])
        plt.xlim([0.0,0.0003])
        plt.axhline(y=0.95, c="purple")
        plt.axvline(x=v95, c="purple")
        plt.axhline(y=0.99)
        plt.axvline(x=v99)
        plt.ylabel('cdf')
        plt.xlabel('set of sampled joint pdf values')

        plt.savefig("cdf.png", dpi=72)

        CS = plt.contour(Z, [v95, v99], colors=("purple", "blue"))
        fmt = {}
        strs = ['5%', '1%']
        for l, s in zip(CS.levels, strs):
                fmt[l] = s 
        #plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10, rightside_up=True)

        #plot max value of the joint pdf
        m = np.argmax(Z)
        i = m % len(Z[0])
        j = (m -i)/ len(Z[0])
        #plt.scatter(i,j, c="red")
        print "analyst location",background.pixelToLatLng(G_Point(i,j))

        #plot other nearby zalando offices
        zalando_offices = [ ((13.446692,52.505535), "Zalando SETamara-Danz-Str. 1, 10243 Berlin"),
                            ((13.41545,52.526460), "Zalando SE Mollstrabe 1, D-10178 Berlin"),
                            ((13.471004,52.506680), "Zalando SE neue bahnhofstrabe 11-17, berlin"),
                            ((13.46323,52.487550), "Zalando SE Am Treptower Park 28 - 30, 12435 Berlin"),
                            ((13.43297,52.505580), "Zalando SE Kopenicker Strbe 20, D-10997 Berlin"),
                            ((13.471004,52.506680), "Zalando Customer Care neue bahnhofstrabe 11-17, berlin"),
                            ((13.436794,52.508725), "Zalando Content Creation Strabe der Pariser Kommune 8  10247 Berlin")]

        offices = [G_LatLng(office[1], office[0]) for office, address in zalando_offices]
        officesPx = map(lambda x: background.latlngToPixel(x), offices)
        X = [t.x for t in officesPx]
        Y = [t.y for t in officesPx]

        plt.clf()
        plt.figure(frameon=False, figsize=(1280/72., 1280/72.), dpi=72)
        plt.axis('off')

        plt.scatter(X,Y, c="#32CD32", s=300)
        plt.scatter(i,j, c="red", s=400)
        CS = plt.contour(Z, [v95, v99], colors=("purple", "blue"), linewidths=5)
        plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=30, rightside_up=True)

        plt.imshow(img,zorder=0)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.savefig("joint.png", dpi=72)

    def main_loop(self, b_map, probs, weights=None):
        N = 10
        N = 30
        X0 = range(2*b_map.mapWidth)
        Y0 = range(2*b_map.mapHeight)
        X = X0[::N] + [X0[-1]]
        Y = Y0[::N] + [Y0[-1]]
        Z_tmp = np.ones((len(X), len(Y)))
        if weights == None:
            weights = [1.0 for i in range(len(probs))]
        for j, y in enumerate(Y):
            for i,x in enumerate(X):
                P = G_Point(x,y)
                for k,prob in enumerate(probs):
                    Z_tmp[j][i] *= weights[k] * prob.prob(P)

        f = interpolate.interp2d(X, Y, Z_tmp, kind='cubic')
        return f(X0,Y0) 

class Gate(object):
    def __init__(self, b_map, center):
	mean = 4.7 #from problem
	mode = 3.877 #from problem
	#mean = exp(mu+((sigma**2)/2))
	self.mu = (1/3.0)*(2.0*np.log(mean)+np.log(mode))
	#mode = exp(mu-sigma**2)
	self.sigma = np.sqrt((2.0/3.0)*(np.log(mean) - np.log(mode)))

        self.b_map = b_map
        self.center = center

    def prob(self, P):
        d = self.b_map.latLngDistance(self.center, self.b_map.pixelToLatLng(P))
        return(np.exp(-(np.log(d) - self.mu)**2 / (2 * self.sigma**2)) / (d * self.sigma * np.sqrt(2 * np.pi)))

class Satellite(object):
    def __init__(self, b_map, S0, S1):
        self.sigma = 2.4/1.96 # 95% falls within mu +/- 1.96*sigma, so sigma == 2.4/1.96
        self.b_map = b_map
	self.s0Px = self.b_map.latlngToPixel(S0)
	self.s1Px = self.b_map.latlngToPixel(S1)

    def prob(self, P):
        nearestPx = self.nearest_point(P, self.s0Px, self.s1Px)
        d = self.b_map.latLngDistance(self.b_map.pixelToLatLng(P), self.b_map.pixelToLatLng(nearestPx))
	return self.gaussian(d, 0.0, self.sigma)

    def gaussian(self, x, mu, sig):
        return (1.0/(sig*np.sqrt(2*np.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def nearest_point(self, P, P0, P1):
        a = P1.y - P0.y
        b = P0.x - P1.x
        c = P1.x*P0.y - P1.y*P0.x
        k = float(a*a + b*b)
        return G_Point((b*(b*P.x-a*P.y)-a*c)/k, (a*(-b*P.x+a*P.y)-b*c)/k)

class River(object):
    def __init__(self, b_map, setup=False):
	self.sigma = 1.39285714286 # 95% falls within mu +/- 1.96*sigma, so sigma == 2.730/1.96
        self.b_map = b_map
	self.points = self.spree_points()
	self.sat = Satellite(self.b_map, G_LatLng(0,0), G_LatLng(0,0))

        if(setup):
            from scipy import spatial
            data = map(lambda x: np.array(x.tolist()), self.points)
            self.tree = spatial.KDTree(data)

    def prob(self, P):
        #use kdtree to find nearest points 
        nearest_points = self.tree.query(np.array(P.tolist()), k=2)[1]
        [P0, P1] = [self.points[x] for x in nearest_points]
	d = self.distance_to_line_segment(P,P0,P1)
	return self.sat.gaussian(d, 0.0, self.sigma)

    def distance_to_line_segment(self, P,P0,P1):
	u = P1 - P0
	v = P - P0

	t = u.dot(v)/u.dot(u) # this is the projection of the point P onto the line segment
	#if 0<=t<=1 the projection lies within the segment and the distance to the line formula can be used, else calculate the distance to the nearest endpoint
        #proj = P0 + t*u
	if t < 0:
                d = self.b_map.latLngDistance(self.b_map.pixelToLatLng(P), self.b_map.pixelToLatLng(P0))
	elif t > 1:
                d = self.b_map.latLngDistance(self.b_map.pixelToLatLng(P), self.b_map.pixelToLatLng(P1))
	elif  t >=0 and t <=1:
                nearestPx = self.sat.nearest_point(P, P0, P1)
                d  = self.b_map.latLngDistance(self.b_map.pixelToLatLng(P), self.b_map.pixelToLatLng(nearestPx))
	return d

    def get_data(self, data_file):
	d = open(data_file).read()
	return d.replace("\n","|")[:-1]

    def spree_points(self):
	data = self.get_data("spree.dat").split("|")
	ans = []
	for p in data:
		tmp = map(float, p.split(","))
                P = G_LatLng(tmp[0], tmp[1])
                PPx = self.b_map.latlngToPixel(P)
                ans.append(PPx)

	return ans

if __name__ == "__main__":
    DensityMap()
