package lakehouse.authz

default allow := false

allow if {
    input.action == "READ"
    input.principal.role == "analytics_agent"
}

allow if {
    input.action == "READ"
    input.principal.role == "contractor_agent"
    input.resource.classification == "PUBLIC"
}
