Pod::Spec.new do |s|
  s.name             = 'flutter_litert_lm'
  s.version          = '0.1.0'
  s.summary          = 'Flutter plugin for on-device LLM inference via Google LiteRT-LM.'
  s.description      = <<-DESC
    A Flutter plugin for on-device LLM inference using Google's LiteRT-LM
    framework. Android is fully supported via the official Kotlin SDK. iOS
    support is pending the official Swift SDK release.
  DESC
  s.homepage         = 'https://github.com/colymba/flutter_litert_lm'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Thierry' => 'thierry@colymba.dev' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform         = :ios, '14.0'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version    = '5.0'
end
