#!/usr/bin/env python

import time, math, heapq, random, re
from collections import deque
import numpy as np
import inkex as ix, simpletransform, simplestyle, cubicsuperpath, cspsubdiv

KICAD_DPI = 96
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

class KiCadExporter(ix.Effect):
  def __init__(self):
    ix.Effect.__init__(self)
    self.OptionParser.add_option('--format', action='store')
    self.OptionParser.add_option('--description', action='store')
    self.OptionParser.add_option('--tags', action='store', default='svg inkscape')
    self.OptionParser.add_option('--flatness', action='store', type='float')
    self.OptionParser.add_option('--tab', action='store')
    self.transformStack = []
    self.currentLayer = ''
    self.polygons = []
    self.module = None

  def effect(self):
    ix.debug('options {}'.format(self.options))
    self.module = [
      MODULE, MODULE_NAME,
      [LAYER, 'F.Cu'],
      [TEDIT, timestamp()],
      [ATTR, 'smd']
    ]

    if self.options.description:
      self.module.append([DESCR, self.options.description])

    if self.options.tags > 0:
      self.module.append([TAGS, self.options.tags])

    doc = self.document.getroot()
    scale = 1 / KICAD_DPI
    scale /= self.unittouu('1px')
    h = self.unittouu(doc.xpath('@height', namespaces=ix.NSS)[0])
    self.push_transform([[scale, 0.0, 0.0], [0.0, -scale, h * scale]])
    self.process_group(doc)
    self.pop_transform()

  def process_group(self, group):
    if group.get(ix.addNS('groupmode', 'inkscape')):
      self.layer = group.get(ix.addNS('label', 'inkscape'))
    trans = group.get('transform')
    if trans:
      self.push_transform(trans)
    for node in group:
      if node.tag == ix.addNS('g', 'svg'):
        self.process_group(node)
      elif node.tag == ix.addNS('use', 'svg'):
        self.process_clone(node)
      else:
        self.process_shape(node, self.current_transform())

      if trans:
        self.pop_transform()

  def process_clone(self, clone):
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
      self.push_transform(mat)

    refid = node.get(ix.addNS('href', 'xlink'))
    refnode = self.getElementById(refid[1:])
    if refnode is not None:
      if refnode.tag == ix.addNS('g', 'svg'):
        self.process_group(refnode)
      elif refnode.tag == ix.addNS('use', 'svg'):
        self.process_clone(refnode)
      else:
        self.process_shape(refnode, self.current_transform())

    if trans or x or y:
      self.pop_transform()

  def process_shape(self, node, mat):
    if node.tag == ix.addNS('path', 'svg'):
      d = node.get('d')
      if not d:
        return
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
    else:
      return

    p = cubicsuperpath.parsePath(d)
    trans = node.get('transform')
    if trans:
      mat = simpletransform.composeTransform(mat, simpletransform.parseTransform(trans))
      simpletransform.applyTransformToPath(mat, p)
    self.construct_polygons(p)

  def construct_polygons(self, p):
    ix.debug('construct_polygons {}'.format(self.layer))
    cspsubdiv.cspsubdiv(p, self.options.flatness)

    rings = []
    for sp in p:
      ring = []
      for csp in sp:
        ring.append([csp[1][0], csp[1][1]])
      ring.pop() # remove duplicate vertex
      rings.append({'inside': [], 'points': ring})

    for r1 in rings:
      for r2 in rings:
        if r1 == r2:
          continue
        (inside, outside) = count_inside(r1['points'], r2['points'])
        if inside != 0 and outside != 0:
          ix.errormsg('Rings of path intersect. outside: {} inside: {}'.format(outside, inside))
        if outside == 0:
          ix.debug('r2 is inside r1')
          r2['inside'].append(r1)

    while len(rings) > 0:
      outer = []
      for r in rings:
        if len(r['inside']) == 0:
          ix.debug('outer ring')
          p = Polygon(self.layer, r['points'])
          r['polygon'] = p
          self.polygons.append(p)
          outer.append(r)
        else:
          ix.debug('inner ring')

      inner = []
      for o in outer:
        for r in rings:
          if o == r:
            continue
          if o in r['inside']:
            if len(r['inside']) == 1:
              points = r['points'] if ring_orientation(r['points']) == CLOCKWISE else r['points'][::-1]
              o['polygon'].innerRings.append(points)
              inner.append(r)
            else:
              r['inside'].remove(o)
        rings.remove(o)
      
      for i in inner:
        for r in rings:
          if i in r['inside']:
            r['inside'].remove(i)
        rings.remove(i)

  def output(self):
    for p in self.polygons:
      self.module.append(kicad_polygon(p))
    print format_sexp(build_sexp(self.module))

  def push_transform(self, t):
    ix.debug('push')
    if isinstance(t, basestring):
      t = simpletransform.parseTransform(t)
    if len(self.transformStack) > 0:
      t = simpletransform.composeTransform(self.current_transform(), t)
    self.transformStack.append(t)

  def current_transform(self):
    return self.transformStack[-1]

  def pop_transform(self):
    ix.debug('pop')
    self.transformStack.pop()
    
def append_ring(poly, ring):
  for p in ring:
    poly.append([XY, "{0:.2f}".format(p[0]), "{0:.2f}".format(p[1])])

def kicad_polygon(p):
  points = [PTS]
  append_ring(points, p.bridge_inner_rings())
  return [FP_POLY, points, [LAYER, p.layer], [WIDTH, 0.0]]

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

def pick_probe_vector(vertices):
  edges = np.roll(vertices, 1, axis = 0) - vertices

  while True:
    direction = 2 * np.pi * random.random()
    ray = np.array([np.cos(direction), np.sin(direction)])
    if 0 not in np.cross(edges, ray):
      return ray

def count_intersections(point, ray, vertices):
  edges = np.roll(vertices, 1, axis = 0) -  vertices

  qs_minus_p = vertices - point
  ray_cross_edges = np.cross(ray, edges)

  ts = np.cross(qs_minus_p, edges) / ray_cross_edges
  if 0 in ts:
    return 1
  us = np.cross(qs_minus_p, ray) / ray_cross_edges
  return np.sum((ts >= 0) & (us >= 0) & (us < 1))

def is_point_in_ring(point, ring):
  return count_intersections(point, pick_probe_vector(ring), ring) % 2 == 1

class Polygon(object):
  def __init__(self, layer, outerRing = []):
    self.outerRing = outerRing
    self.innerRings = []
    self.layer = layer

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
        ix.debug('found bridge {}'.format(idx))
        return idx
    return None

  def insert_bridged_hole(self, result, hole, bridge):
    [oi, ii] = bridge
    h = deque(hole)
    h.rotate(ii)
    h.append(h[0])
    h.append(result[oi])
    result[oi+1:oi+1] = h

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

# given three colinear points p, q, r check if q lies on the line segment pr
def on_segment(p, q, r):
  for i in range(len(p)):
    # XXX
    # if not (q[i] <= max(p[i], r[i]) and q[i] >= min(p[i], r[i])):
    if not (q[i] < max(p[i], r[i]) and q[i] > min(p[i], r[i])):
      return False
  return True


# find orientation of ordered triple (p, q, r)
def orientation(p, q, r):
  val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
  if val < EPSILON:
    return COLINEAR
  return CLOCKWISE if val > 0 else COUNTERCLOCKWISE # clock or counterclockwise

# return wether line segment p1q1 intersects p2q2
def do_intersect(p1, q1, p2, q2):
  o1 = orientation(p1, q1, p2)
  o2 = orientation(p1, q1, q2)
  o3 = orientation(p2, q2, p1)
  o4 = orientation(p2, q2, q1)

  if o1 != o2 and o3 != o4:
    return True

  if o1 == COLINEAR and on_segment(p1, p2, q1):
    return True

  if o2 == COLINEAR and on_segment(p1, q2, q1):
    return True

  if o3 == COLINEAR and on_segment(p2, p1, q2):
    return True

  if o4 == COLINEAR and on_segment(p2, q1, q2):
    return True

  return False


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
  KiCadExporter().affect()

