"""
SELECT ?xlabel ?ylabel ?zlabel ?text where
{
    ?x ?y ?z .
    { ?x rdf:type dbo:Place }
    { ?z rdf:type dbo:Place }

    FILTER (?x = <http://dbpedia.org/resource/Headingley>)

    ?x rdfs:label ?xlabel . FILTER (lang(?xlabel) = 'en')
    ?x dbo:abstract ?text . FILTER (lang(?text) = 'en')
    ?y rdfs:label ?ylabel . FILTER (lang(?ylabel) = 'en')
    ?z rdfs:label ?zlabel . FILTER (lang(?zlabel) = 'en')
}
"""
