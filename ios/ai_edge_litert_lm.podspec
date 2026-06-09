Pod::Spec.new do |s|
  s.name             = 'ai_edge_litert_lm'
  s.version          = '0.1.0'
  s.summary          = 'Flutter plugin for on-device LLM inference via Google LiteRT-LM.'
  s.description      = <<-DESC
    A Flutter plugin for on-device LLM inference using Google's LiteRT-LM
    framework. Android and iOS are fully supported via the official SDKs.
  DESC
  s.homepage         = 'https://github.com/colymba/ai_edge_litert_lm'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Thierry' => 'thierry@colymba.dev' }
  s.source           = { :path => '.' }
  s.source_files     = 'ai_edge_litert_lm/Sources/ai_edge_litert_lm/**/*'
  s.dependency 'Flutter'
  s.platform         = :ios, '15.0'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version    = '5.0'
end
