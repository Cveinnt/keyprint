import json
import pytest
from keyprint import Comparison, export_gallery


def pair(text='🌱'):
    row={'text':text,'completion':'eos','inspection':{'series':[]},
         'private_key':'SECRET','trace':[dict(index=0,token_id=1,bytes_hex='41',
             text=text,start=0,end=len(text),kind='text',random_draw='SECRET')]}
    return Comparison('Prompt',dict(outputs={'ordinary':row,'marked':row},
                                   seconds=1,max_tokens=32,private_key='SECRET'))


def test_gallery_reuses_viewer_and_preserves_all_results(tmp_path):
    entry=export_gallery({'One':pair('First'),'Deux':pair('Deuxième')},tmp_path/'gallery',
                         notes={'Deux':'Retained failure; not approved.'})
    assert 'data-gallery="true"' in entry.read_text()
    assert (entry.parent/'gallery.js').is_file()
    data=json.loads((entry.parent/'gallery.json').read_text())
    assert [e['title'] for e in data['examples']]==['One','Deux']
    assert data['examples'][1]['recording']['experiment']['outputs']['marked']['text']=='Deuxième'
    assert data['examples'][1]['note']=='Retained failure; not approved.'
    assert 'SECRET' not in json.dumps(data)
    assert json.loads((entry.parent/'replay.json').read_text())==data['examples'][0]['recording']
    with pytest.raises(FileExistsError):export_gallery({'One':pair()},entry.parent)


@pytest.mark.parametrize('examples,error',[
    ({},ValueError),({'':pair()},ValueError),({'a'*81:pair()},ValueError),
    ({str(i):pair() for i in range(13)},ValueError),({'One':{}},TypeError),
])
def test_bad_gallery_creates_nothing(tmp_path,examples,error):
    target=tmp_path/'gallery'
    with pytest.raises(error):export_gallery(examples,target)
    assert not target.exists()


@pytest.mark.parametrize('notes',[{'Missing':'note'},{'One':None},{'One':'a'*1001},[]])
def test_bad_notes_create_nothing(tmp_path,notes):
    with pytest.raises(ValueError):export_gallery({'One':pair()},tmp_path/'site',notes=notes)
    assert not (tmp_path/'site').exists()


def test_export_preserves_only_known_backend_names():
    item=pair();item.result['backend']='transformers'
    assert item.to_dict()['experiment']['backend']=='transformers'
    item.result['backend']='/private/path/SECRET'
    assert item.to_dict()['experiment']['backend'] is None
