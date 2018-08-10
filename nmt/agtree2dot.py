#########################################################################
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the version 3 of the GNU General Public License #
# as published by the Free Software Foundation.                         #
#                                                                       #
# This program is distributed in the hope that it will be useful, but   #
# WITHOUT ANY WARRANTY; without even the implied warranty of            #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      #
# General Public License for more details.                              #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program. If not, see <http://www.gnu.org/licenses/>.  #
#                                                                       #
# Written by and Copyright (C) Francois Fleuret                         #
# Contact <francois.fleuret@idiap.ch> for comments & bug reports        #
#########################################################################

import torch
import sys, re

######################################################################

class Link:
    def __init__(self, from_node, from_nb, to_node, to_nb):
        self.from_node = from_node
        self.from_nb = from_nb
        self.to_node = to_node
        self.to_nb = to_nb

class Node:
    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.max_in = -1
        self.max_out = -1

def slot(node_list, n, k, for_input):
    if for_input:
        if node_list[n].max_out > 0:
            return str(node_list[n].id) + ':input' + str(k)
        else:
            return str(node_list[n].id)
    else:
        if node_list[n].max_in > 0:
            return str(node_list[n].id) + ':output' + str(k)
        else:
            return str(node_list[n].id)

def slot_string(k, for_input):
    result = ''

    if for_input:
        label = 'input'
    else:
        label = 'output'

    if k > 0:
        if not for_input: result = ' |' + result
        result +=  ' { <' + label + '0> 0'
        for j in range(1, k + 1):
            result += " | " + '<' + label + str(j) + '> ' + str(j)
        result += " } "
        if for_input: result = result + '| '

    return result

######################################################################

def add_link(node_list, link_list, u, nu, v, nv):
    if u is not None and v is not None:
        link = Link(u, nu, v, nv)
        link_list.append(link)
        node_list[u].max_in  = max(node_list[u].max_in,  nu)
        node_list[v].max_out = max(node_list[v].max_out, nv)

######################################################################

def fill_graph_lists(u, node_labels, node_list, link_list):

    if u is not None and not u in node_list:
        node = Node(len(node_list) + 1,
                    (u in node_labels and node_labels[u]) or \
                    re.search('<class \'(.*\.|)([a-zA-Z0-9_]*)\'>', str(type(u))).group(2))
        node_list[u] = node

        if hasattr(u, 'grad_fn'):
            fill_graph_lists(u.grad_fn, node_labels, node_list, link_list)
            add_link(node_list, link_list, u, 0, u.grad_fn, 0)

        if hasattr(u, 'variable'):
            fill_graph_lists(u.variable, node_labels, node_list, link_list)
            add_link(node_list, link_list, u, 0, u.variable, 0)

        if hasattr(u, 'next_functions'):
            for i, (v, j) in enumerate(u.next_functions):
                fill_graph_lists(v, node_labels, node_list, link_list)
                add_link(node_list, link_list, u, i, v, j)

######################################################################

def print_dot(node_list, link_list, out):
    out.write('digraph{\n')

    for n in node_list:
        node = node_list[n]

        if isinstance(n, torch.autograd.Variable):
            out.write(
                '  ' + \
                str(node.id) + ' [shape=note,style=filled, fillcolor="#e0e0ff",label="' + \
                node.label + ' ' + re.search('torch\.Size\((.*)\)', str(n.data.size())).group(1) + \
                '"]\n'
            )
        else:
            out.write(
                '  ' + \
                str(node.id) + ' [shape=record,style=filled, fillcolor="#f0f0f0",label="{ ' + \
                slot_string(node.max_out, for_input = True) + \
                node.label + \
                slot_string(node.max_in, for_input = False) + \
                ' }"]\n'
            )

    for n in link_list:
        out.write('  ' + \
                  slot(node_list, n.from_node, n.from_nb, for_input = False) + \
                  ' -> ' + \
                  slot(node_list, n.to_node, n.to_nb, for_input = True) + \
                  '\n')

    out.write('}\n')

######################################################################

def save_dot(x, node_labels = {}, out = sys.stdout):
    node_list, link_list = {}, []
    fill_graph_lists(x, node_labels, node_list, link_list)
    print_dot(node_list, link_list, out)

######################################################################
