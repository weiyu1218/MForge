{{- define "moleculeforge.labels" -}}
app.kubernetes.io/part-of: moleculeforge
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end -}}
