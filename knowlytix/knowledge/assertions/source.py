# SPDX-License-Identifier: Apache-2.0
"""Lossless source occurrences and bounded structural context; no model imports."""
from __future__ import annotations

import hashlib
import json
import re

VERSION = 'assertions/1'


def anchored_text(value):
    """Render a quoted phrase, including source parts interrupted by footnotes."""
    if 'phrase' in value:
        return value['phrase']['text']
    if 'mention' in value:
        return ' '.join([value['referent']['text'], *[m['text'] for m in value['modifiers']]])
    return ' '.join(p['text'] for p in value['parts']) if 'parts' in value else value['text']


def role_structure(value):
    """Preserve reference identity and attachment in semantic scoring."""
    if 'phrase' in value:
        return {'phrase': value['phrase']['text'], 'bindings': [
            {'mention': b['mention']['text'], 'relation': b['relation'], 'target': b['target']['text']}
            for b in value['bindings']]}
    if 'mention' in value:
        return {'mention': value['mention']['text'], 'referent': value['referent']['text'],
                'modifiers': [m['text'] for m in value['modifiers']]}
    return anchored_text(value)


def unresolved_references(references):
    """Bibliographic attribution is provenance, not missing rule content."""
    return [r for r in references if r['state'] != 'resolved' and r.get('usage') != 'citation']


def bibliographic_reference(text, label):
    """Recognize bibliography entries and explicit further-reading pointers.

    If the same label also appears in substantive prose it remains blocking.
    """
    matches = list(re.finditer(re.escape(label) + r'(?!\w)', text, re.I))
    for match in matches:
        prefix = text[text.rfind('\n', 0, match.start()) + 1:match.start()]
        bibliography = re.match(r'^\s*\d+\s*\w', prefix) and re.search(r'[,;]\s*["“]', prefix)
        pointer = re.search(r'\bRefer to\s*$', prefix, re.I) and re.match(
            r'(?:\s*\([^\n]*?\))?\s+for additional (?:guidance|information)\s*\.', text[match.end():], re.I)
        if not (bibliography or pointer):
            return False
    return bool(matches)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def anchor(document, start, end):
    return {'document': document['id'], 'start': start, 'end': end,
            'text': document['text'][start:end]}


def index_documents(documents, *, reference_registry=None):
    """Index explicit document identities. Offsets are Unicode code points.

    Primary units partition nonblank content. Headings/separators are structural;
    every other primary unit requires an extraction disposition, without a modal
    keyword filter. Duplicate headings and repeated occurrences keep distinct IDs.
    """
    reference_registry = reference_registry or {}
    if not isinstance(reference_registry, dict):
        raise ValueError('reference registry must map mentions to explicit document/heading targets')
    for label, target in reference_registry.items():
        if (not isinstance(label, str) or not isinstance(target, dict)
                or set(target) != {'document', 'heading'}
                or not all(isinstance(v, str) and v for v in target.values())):
            raise ValueError('invalid reference registry entry')
    indexed, units = [], []
    names = set()
    for raw in documents:
        name, text = raw['name'], raw['text']
        if not isinstance(name, str) or not name or name in names or not isinstance(text, str):
            raise ValueError('documents require unique names and string text')
        names.add(name)
        if raw.get('format', 'markdown') not in {'markdown', 'text'}:
            raise ValueError('unsupported document format; supply a tested text adapter')
        doc = {'id': digest([name, text]), 'name': name, 'text': text,
               'sha256': hashlib.sha256(text.encode()).hexdigest(),
               'format': raw.get('format', 'markdown'), 'language': raw.get('language', 'en')}
        indexed.append(doc)
        headings, lists, paragraph = [], [], []
        previous, table_header, paragraph_parent = None, None, None

        def add(kind, start, end, parent=None, links=()):
            nonlocal previous
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            value = {'id': digest([doc['id'], kind, start, end]), 'kind': kind,
                     'source': anchor(doc, start, end), 'parent': parent,
                     'links': list(links), 'structural': kind in {'heading', 'table_separator'}}
            units.append(value)
            if not value['structural']:
                previous = value
            return value

        def flush():
            if paragraph:
                body = text[paragraph[0]:paragraph[-1]]
                kind = 'caption' if re.match(r'^(?:Table|Figure)\s+\d+[.:]', body) else 'paragraph'
                add(kind, paragraph[0], paragraph[-1], paragraph_parent)
                paragraph.clear()

        offset = 0
        for line in text.splitlines(keepends=True):
            lo, hi = offset, offset + len(line)
            offset = hi
            body = line.rstrip('\r\n')
            heading = re.match(r'^(#{1,6})\s+(.+)', body) if doc['format'] == 'markdown' else None
            bullet = re.match(r'^(\s*)(?:[-*+] |\d+[.)] )', body)
            footnote = re.match(r'^\[\^[^]]+\]:', body)
            table = body.lstrip().startswith('|') and body.count('|') >= 2
            if not body.strip():
                flush()
                table_header = None
                continue
            if heading:
                flush()
                level = len(heading[1])
                while headings and headings[-1][0] >= level:
                    headings.pop()
                unit = add('heading', lo, hi, headings[-1][1] if headings else None)
                headings.append((level, unit['id']))
                lists, previous, table_header = [], None, None
            elif footnote or bullet or table:
                flush()
                parent = headings[-1][1] if headings else None
                links = []
                if footnote:
                    kind = 'footnote'
                elif table:
                    kind = 'table_separator' if re.fullmatch(r'[\s|:\-]+', body) else 'table_row'
                    if table_header:
                        links.append({'kind': 'table_header', 'target': table_header})
                    lists = []
                else:
                    kind, indent = 'list_item', len(bullet[1].expandtabs(4))
                    sibling = None
                    while lists and lists[-1][0] >= indent:
                        level, popped = lists.pop()
                        if level == indent:
                            sibling = next(u for u in reversed(units) if u['id'] == popped)
                    if lists:
                        parent = lists[-1][1]
                        links.append({'kind': 'list_parent', 'target': parent})
                    elif sibling is not None:
                        parent = sibling['parent']
                        if parent:
                            links.append({'kind': 'list_introduction', 'target': parent})
                    elif previous and previous['source']['text'].rstrip().endswith(':'):
                        parent = previous['id']
                        links.append({'kind': 'list_introduction', 'target': parent})
                    elif previous and previous['kind'] == 'list_item':
                        parent = previous['parent']
                        if parent:
                            links.append({'kind': 'list_introduction', 'target': parent})
                    table_header = None
                unit = add(kind, lo, hi, parent, links)
                if bullet and not footnote and not table:
                    lists.append((indent, unit['id']))
                if table and table_header is None and kind == 'table_row':
                    table_header = unit['id']
            else:
                # Indented continuation/footnote text must not terminate an open
                # list. A later sibling still inherits the same governing item.
                if not body[:1].isspace():
                    lists = []
                table_header = None
                if not paragraph:
                    paragraph_parent = (previous['id'] if body[:1].isspace() and previous and previous['kind'] in {'list_item', 'footnote'}
                                        else headings[-1][1] if headings else None)
                    paragraph.append(lo)
                paragraph.append(hi)
        flush()
    # Explicit heading and footnote references retain unresolved/multiple targets.
    for unit in units:
        if unit['structural']:
            continue
        source = unit['source']
        for match in re.finditer(r'\[\^(?P<foot>[^]]+)\]|\b(?P<ref>(?:Appendix|Section|Chapter)\s+[\w.\-]+)', source['text'], re.I):
            label = (match['ref'].rstrip('.') if match['ref'] else None) or f"[^{match['foot']}]"
            targets = [u['id'] for u in units if u['source']['document'] == source['document']
                       and (u['kind'] == 'footnote' and match['foot'] and u['source']['text'].startswith(label + ':')
                            or u['kind'] == 'heading' and match['ref']
                            and re.sub(r'^#+\s*', '', u['source']['text']).casefold().rstrip(':') == label.casefold())]
            registered = reference_registry.get(label)
            if match['ref']:
                identifier = label.split()[-1]
                formal = re.fullmatch(r'(?:\d[\w.\-]*|[A-Z]|[IVXLCDM]+)', identifier, re.I)
                # Do not mistake prose such as "Section to be updated" for a
                # reference. Keep named sections/explicit registrations too.
                if not (formal or identifier[:1].isupper() or targets or registered):
                    continue
            if registered is not None:
                target_docs = {d['id'] for d in indexed if d['name'] == registered['document']}
                targets = [u['id'] for u in units if u['source']['document'] in target_docs
                           and u['kind'] == 'heading'
                           and re.sub(r'^#+\s*', '', u['source']['text']).casefold().rstrip(':') == registered['heading'].casefold()]
            link = {'kind': 'reference', 'mention': label, 'targets': targets,
                    'state': 'resolved' if len(targets) == 1 else 'missing' if not targets else 'ambiguous'}
            # A numbered bibliographic line quoting a section title is not a
            # directive to follow that section. Keep the link for provenance.
            if match['ref'] and bibliographic_reference(source['text'], label):
                link['usage'] = 'citation'
            unit['links'].append(link)
    fragments = []
    for unit in units:
        source = unit['source']
        if unit['structural']:
            continue
        spans = []
        if unit['kind'] == 'table_row':
            boundaries = [m.start() for m in re.finditer(r'(?<!\\)\|', source['text'])]
            spans = [('table_cell', a + 1, b) for a, b in zip(boundaries, boundaries[1:])]
        elif unit['kind'] in {'paragraph', 'list_item', 'footnote', 'caption'}:
            starts = [0, *[m.end() for m in re.finditer(r'[.!?]\s+(?=[A-Z])', source['text'])]]
            spans = [('sentence', a, b) for a, b in zip(starts, starts[1:] + [len(source['text'])])]
        for kind, lo, hi in spans:
            while lo < hi and source['text'][lo].isspace():
                lo += 1
            while hi > lo and source['text'][hi - 1].isspace():
                hi -= 1
            if lo == hi:
                continue
            fragment_source = {**source, 'start': source['start'] + lo, 'end': source['start'] + hi,
                               'text': source['text'][lo:hi]}
            fragments.append({'id': digest([unit['id'], kind, lo, hi]), 'parent': unit['id'],
                              'kind': kind, 'source': fragment_source,
                              'boundary_method': 'text-boundaries-v1'})
    return {'version': VERSION, 'documents': indexed, 'units': units,
            'fragments': fragments, 'reference_registry': reference_registry}


def assemble_context(index, unit_id, *, max_chars=16000, max_units=12):
    if max_chars < 1 or max_units < 1:
        raise ValueError('context limits must be positive')
    units = {u['id']: u for u in index['units']}
    queue, selected, missing = [unit_id], [], []
    seen = set()
    target = units[unit_id]
    if target['source']['text'].endswith(':'):
        queue.extend(u['id'] for u in index['units'] if u['parent'] == unit_id)
    queue.extend(u['id'] for u in index['units'] if any(
        link.get('kind') == 'table_header' and link.get('target') == unit_id
        for link in u['links']))
    if re.search(r'\b(it|they|this|these|those|such)\b', target['source']['text'], re.I):
        previous = [u for u in index['units'] if not u['structural']
                    and u['source']['document'] == target['source']['document']
                    and u['parent'] == target['parent'] and u['source']['end'] <= target['source']['start']]
        if previous:
            queue.append(previous[-1]['id'])
    while queue:
        key = queue.pop(0)
        if key in seen:
            continue
        seen.add(key)
        unit = units[key]
        if len(selected) >= max_units or sum(len(u['source']['text']) for u in selected) + len(unit['source']['text']) > max_chars:
            missing.append({'kind': 'context_budget', 'unit': key})
            continue
        selected.append(unit)
        # A definition/list introduction is incomplete without its children,
        # including when reached through a child or an anaphoric reference.
        if unit['source']['text'].endswith(':'):
            queue.extend(u['id'] for u in index['units'] if u['parent'] == key)
        if unit['parent']:
            queue.append(unit['parent'])
        for link in unit['links']:
            if link['kind'] == 'reference':
                if link.get('usage') == 'citation':
                    continue
                if link['state'] == 'resolved':
                    queue.extend(link['targets'])
                    # A section reference needs the section contents, not only its title.
                    descendants = set(link['targets'])
                    changed = True
                    while changed:
                        before = len(descendants)
                        descendants.update(u['id'] for u in index['units'] if u['parent'] in descendants)
                        changed = len(descendants) != before
                    queue.extend(u['id'] for u in index['units'] if u['id'] in descendants)
                else:
                    missing.append(link)
            elif link.get('target'):
                queue.append(link['target'])
    return {'unit': unit_id, 'units': selected, 'missing': missing,
            'digest': digest([selected, missing])}
