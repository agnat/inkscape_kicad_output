#!/usr/bin/env python

import sys, time, math, heapq, re
from collections import deque
import numpy as np
import inkex as ix, simpletransform, simplestyle, cubicsuperpath, cspsubdiv

MODULE_NAME = 'inkscape-kicad-output'

COLINEAR = 0
CLOCKWISE = 1
COUNTERCLOCKWISE = 2
EPSILON = 1e-6

# KiCad Mod S-Expression Constants
MODULE = 'module'
LAYER = 'layer'
TEDIT = 'tedit'
ATTR = 'attr'
DESCR = 'descr'
TAGS = 'tags'
FP_POLY = 'fp_poly'
PTS = 'pts'
XY = 'xy'
WIDTH = 'width'

#==============================================================================

class KiCadOutput(ix.Effect):
  def __init__(self):
    ix.Effect.__init__(self)
    self.OptionParser.add_option('--format', action='store')
    self.OptionParser.add_option('--layer-mode', action='store')
    self.OptionParser.add_option('--target-layer', action='store')
    self.OptionParser.add_option('--description', action='store')
    self.OptionParser.add_option('--tags', action='store', default='svg inkscape')
    self.OptionParser.add_option('--flatness', action='store', type='float')
    self.transformStack = []
    self.layer = ''
    self.builder = None

    self.polygons = []
    self.module = None

  def effect(self):
    # ix.debug('options: {}'.format(self.options))
    if self.options.format == 'footprint':
      self.builder = KiCadFootprintBuilder(self.options)
    else:
      abort('Unhandled format "{}"'.format(self.options.format))

    self.module = [
      MODULE, MODULE_NAME,
      [LAYER, 'F.Cu'],
      [TEDIT, timestamp()],
      [ATTR, 'smd']
    ]

    if self.options.layer_mode == 'target':
      self.layer = self.options.target_layer

    if self.options.description:
      self.module.append([DESCR, self.options.description])

    if self.options.tags:
      self.module.append([TAGS, self.options.tags])

    doc = self.document.getroot()
    scale = 1 / self.unittouu('1mm')
    self.pushTransform([[scale, 0.0, 0.0], [0.0, scale, 0.0]])
    # h = self.unittouu(doc.xpath('@height', namespaces=ix.NSS)[0])
    # self.pushTransform([[scale, 0.0, 0.0], [0.0, scale, h * scale]])
    self.processGroup(doc)
    self.popTransform()

  def output(self):
    print self.builder.output()

  def processGroup(self, group):
    if self.options.layer_mode == 'document' and group.get(ix.addNS('groupmode', 'inkscape')):
      self.layer = group.get(ix.addNS('label', 'inkscape'))

    trans = group.get('transform')
    if trans:
      self.pushTransform(trans)
    
    for node in group:
      if node.tag == ix.addNS('g', 'svg'):
        self.processGroup(node)
      elif node.tag == ix.addNS('use', 'svg'):
        self.processClone(node)
      else:
        self.processShape(node, self.currentTransform())

    if trans:
      self.popTransform()

  def processClone(self, clone):
    trans = node.get('transform')
    x = node.get('x')
    y = node.get('y')
    mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    if trans:
      mat = simpletransform.composeTransform(mat, simpletransform.parseTransform(trans))
    if x:
      mat = simpletransform.composeTransform(mat, [[1.0, 0.0, float(x)], [0.0, 1.0, 0.0]])
    if y:
      mat = simpletransform.composeTransform(mat, [[1.0, 0.0, 0.0], [0.0, 1.0, float(y)]])

    if trans or x or y:
      self.pushTransform(mat)

    refid = node.get(ix.addNS('href', 'xlink'))
    refnode = self.getElementById(refid[1:])
    if refnode is not None:
      if refnode.tag == ix.addNS('g', 'svg'):
        self.processGroup(refnode)
      elif refnode.tag == ix.addNS('use', 'svg'):
        self.processClone(refnode)
      else:
        self.processShape(refnode, self.currentTransform())

    if trans or x or y:
      self.popTransform()

  def processShape(self, node, mat):
    d = None
    if node.tag == ix.addNS('path', 'svg'):
      d = node.get('d')
    elif node.tag == ix.addNS('rect', 'svg'):
      x = float(node.get('x', 0))
      y = float(node.get('y', 0))
      width = float(node.get('width'))
      height = float(node.get('height'))
      d = "m %s,%s %s,%s %s,%s %s,%s z".format(x, y, width, 0, 0, height, -width, 0)
    elif node.tag == ix.addNS('line', 'svg'):
      x1 = float(node.get('x1', 0))
      x2 = float(node.get('x2', 0))
      y1 = float(node.get('y1', 0))
      y2 = float(node.get('y2', 0))
      d = "M %s,%s L %s,%s".format(x1, y1, x2, y2)
    elif node.tag == ix.addNS('circle', 'svg'):
      cx = float(node.get('cx', 0))
      cy = float(node.get('cy', 0))
      r = float(node.get('r'))
      d = "m %s,%s a %s,%s 0 0 1 %s,%s %s,%s 0 0 1 %s,%s z".format(cx + r, cy, r, r, -2*r, 0, r, r, 2*r, 0)
    elif node.tag == ix.addNS('ellipse','svg'):
      cx = float(node.get('cx', 0))
      cy = float(node.get('cy', 0))
      rx = float(node.get('rx'))
      ry = float(node.get('ry'))
      d = "m %s,%s a %s,%s 0 0 1 %s,%s %s,%s 0 0 1 %s,%s z".format(cx + rx, cy, rx, ry, -2*rx, 0, rx, ry, 2*rx, 0)

    if not d:
      return

    p = cubicsuperpath.parsePath(d)
    trans = node.get('transform')
    if trans:
      self.pushTransform(trans)
    simpletransform.applyTransformToPath(self.currentTransform(), p)
    self.builder.appendPolygonsFromPath(p, self.layer)
    if trans:
      self.popTransform()

  def pushTransform(self, t):
    if not isinstance(t, list):
      t = simpletransform.parseTransform(t)
    if len(self.transformStack) > 0:
      t = simpletransform.composeTransform(self.currentTransform(), t)
    self.transformStack.append(t)

  def currentTransform(self):
    return self.transformStack[-1] if len(self.transformStack) > 0 else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

  def popTransform(self):
    self.transformStack.pop()
    
#==============================================================================


class KiCadBuilder(object):
  def __init__(self, options):
    self.options = options
    self.expression = []

  def output(self):
    return format_sexp(build_sexp(self.expression))

  def appendPolygonsFromPath(self, path, layer):
    cspsubdiv.cspsubdiv(path, self.options.flatness)

    polygons = constructBridgedPolygonsFromPath(path)

    for polygon in polygons:
      self.appendPolygon(polygon, layer)


class KiCadFootprintBuilder(KiCadBuilder):
  def __init__(self, options):
    super(KiCadFootprintBuilder, self).__init__(options)
    self.expression = [
      MODULE, MODULE_NAME,
      [LAYER, 'F.Cu'],
      [TEDIT, timestamp()],
      [ATTR, 'smd']
    ]
    if options.description:
      self.expression.append([DESCR, options.description])

    if options.tags > 0:
      self.expression.append([TAGS, options.tags])

  def appendPolygon(self, polygon, layer, width = 0.0):
    points = [PTS]
    points.extend([[XY, p[0], p[1]] for p in polygon])
    self.expression.append([FP_POLY, points, [LAYER, layer], [WIDTH, width]])


#==============================================================================

def extractRings(path):
  rings = []
  for sp in path:
    ring = []
    for csp in sp:
      ring.append([csp[1][0], csp[1][1]])
    ring.pop() # remove duplicate vertex
    rings.append(Ring(ring))
  return rings

def probeRingContainment(rings):
  for r1 in rings:
    for r2 in rings:
      if r1 == r2:
        continue
      (inside, outside) = count_inside(r1.points, r2.points)
      if inside != 0 and outside != 0:
        # TODO: Provide context
        abort('Rings of path intersect. outside: {} inside: {}'.format(outside, inside))
      if outside == 0:
        r2.containedIn.append(r1)


def peelPolygons(rings):
  polygons = []
  while len(rings) > 0:
    startLength = len(rings)
    outer = []
    for r in rings:
      if len(r.containedIn) == 0:
        r.points = r.points if ring_orientation(r.points) == COUNTERCLOCKWISE else r.points[::-1]
        polygon = Polygon(r.points)
        r.polygon = polygon
        polygons.append(polygon)
        outer.append(r)

    inner = []
    for o in outer:
      for r in rings:
        if o == r:
          continue
        if o in r.containedIn:
          if len(r.containedIn) == 1:
            points = r.points if ring_orientation(r.points) == CLOCKWISE else r.points[::-1]
            o.polygon.innerRings.append(points)
            inner.append(r)
          else:
            r.containedIn.remove(o)
      rings.remove(o)
    
    for i in inner:
      for r in rings:
        if i in r.containedIn:
          r.containedIn.remove(i)
      rings.remove(i)

    if startLength == len(rings):
      abort('Didn\'t make any progress. Kaputt.')
  return polygons


def constructBridgedPolygonsFromPath(path):
  rings = extractRings(path)
  probeRingContainment(rings)
  polygons = peelPolygons(rings)
  return [p.bridge_inner_rings() for p in polygons]

#==============================================================================

def timestamp():
  return "{0:8X}".format(int(time.time()))

# compute the signed area of a given simple polygon
# http://mathworld.wolfram.com/PolygonArea.html
def signed_area(ring):
  area = 0
  n = len(ring)
  for i in range(n):
    nxt = (i + 1) % n
    area += ring[i][0] * ring[nxt][1] - ring[nxt][0] * ring[i][1]
  return 0.5 * area

# find orientation of a simple polygon
def ring_orientation(ring):
  return COUNTERCLOCKWISE if signed_area(ring) > 0 else CLOCKWISE

def count_inside(r1, r2):
  inside = 0
  outside = 0
  for p in r2:
    if is_point_in_ring(np.array(p), np.array(r1)):
      inside += 1
    else:
      outside += 1
  return (inside, outside)

# https://en.wikipedia.org/wiki/Even-odd_rule
def is_point_in_ring(p, poly):
  """
  p -- a point
  poly -- a list of tuples [(x, y), (x, y), ...]
  """
  x = p[0]
  y = p[1]
  num = len(poly)
  i = 0
  j = num - 1
  c = False
  for i in range(num):
    if ((poly[i][1] > y) != (poly[j][1] > y)) and \
       (x < poly[i][0] + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
       (poly[j][1] - poly[i][1])):
      c = not c
    j = i
  return c

class Polygon(object):
  def __init__(self, outerRing = []):
    self.outerRing = outerRing
    self.innerRings = []

  # bridge outer ring and holes: https://arxiv.org/pdf/1212.6038.pdf
  def bridge_inner_rings(self):
    result = self.outerRing[:]
    for hole in self.innerRings:
      bridge = self.select_bridge(result, hole)
      self.insert_bridged_hole(result, hole, bridge)
    return result

  def select_bridge(self, outer, inner):
    candidates = []
    for (i, pi) in enumerate(inner):
      for (o, po) in enumerate(outer):
        heapq.heappush(candidates, (length(po, pi), [po, pi], [o, i]))
    while len(candidates) > 0:
      (l, e, idx) = heapq.heappop(candidates)
      if not edge_intersects_ring(e, outer) and not edge_intersects_ring(e, inner):
        return idx
    return None

  def insert_bridged_hole(self, result, hole, bridge):
    [oi, ii] = bridge
    h = deque(hole)
    h.rotate(ii)
    h.append(h[0])
    h.append(result[oi])
    result[oi+1:oi+1] = h

class Ring(object):
  def __init__(self, points):
    self.points = points
    self.polygon = None
    self.containedIn = []

def edge_intersects_ring(e, r):
  n = len(r)
  for (i, p0) in enumerate(r):
    nxt = (i + 1) % n
    p1 = r[nxt]
    if do_intersect(p0, p1, e[0], e[1]):
      return True
  return False

def length(p1, p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx * dx + dy * dy)

# find orientation of ordered triple (p, q, r)
def orientation(p, q, r):
  val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
  if val < EPSILON:
    return COLINEAR
  return CLOCKWISE if val > 0 else COUNTERCLOCKWISE

# return wether line segment p1q1 intersects p2q2
def do_intersect(p1, q1, p2, q2):
  o1 = orientation(p1, q1, p2)
  o2 = orientation(p1, q1, q2)
  o3 = orientation(p2, q2, p1)
  o4 = orientation(p2, q2, q1)

  return o1 != o2 and o3 != o4

def abort(message):
  ix.errormsg(message)
  sys.exit(1)

# From https://github.com/KiCad/kicad-library-utils
# kicad-library-utils/common/sexpr.py

float_render = "%.2f"

term_regex = r'''(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>[+-]?\d+\.\d+(?=[\ \)])|\-?\d+(?=[\ \)]))|
        (?P<sq>"([^"]|(?<=\\)")*")|
        (?P<s>[^(^)\s]+)
       )'''

def build_sexp(exp, key=None):
  out = ''
  
  # Special case for multi-values
  if type(exp) == type([]):
    out += '('+ ' '.join(build_sexp(x) for x in exp) + ')'
    return out
  elif type(exp) == type('') and re.search(r'[\s()]', exp):
    out += '"%s"' % repr(exp)[1:-1].replace('"', r'\"')
  elif type(exp) in [int,float]:
    out += float_render % exp
  else:
    if exp == '':
      out += '""'
    else:
      out += '%s' % exp
  
  if key is not None:
    out = "({key} {val})".format(key=key, val=out)
      
  return out

def format_sexp(sexp, indentation_size=2, max_nesting=2):
  out = ''
  n = 0
  for termtypes in re.finditer(term_regex, sexp):
    indentation = ''
    term, value = [(t,v) for t,v in termtypes.groupdict().items() if v][0]
    if term == 'brackl':
      if out:
        if n <= max_nesting:
          if out[-1] == ' ':
            out = out[:-1]
            indentation = '\n' + (' ' * indentation_size * n)
          else:
            if out[-1] == ')': out += ' '
      n += 1
    elif term == 'brackr':
      if out and out[-1] == ' ': out = out[:-1]
      n -= 1
    elif term == 'num':
      value += ' '
    elif term == 'sq':
      value += ' '
    elif term == 's':
      value += ' '
    else:
      raise NotImplementedError("Error: %r" % (term, value))

    out += indentation + value

  out += '\n'
  return out


if __name__ == '__main__':
  ix.localize()
  KiCadOutput().affect()

